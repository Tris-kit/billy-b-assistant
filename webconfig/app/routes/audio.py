import configparser
import glob
import json
import os
import queue
import re
import subprocess
import time

import numpy as np
import paho.mqtt.client as mqtt
import sounddevice as sd
from flask import Blueprint, Response, jsonify, request, send_from_directory

from core.wakeup import generate_wake_clip_async

from ..core_imports import core_config
from ..state import PERSONA_PATH, PROJECT_ROOT, WAKE_UP_DIR


bp = Blueprint("audio", __name__)

mic_check_running = False
rms_queue = queue.Queue()


def _is_selectable_audio_device_name(name: str) -> bool:
    """Keep real hardware endpoints (USB, Bluetooth, built-in) and skip virtual PCMs."""
    lowered = (name or "").strip().lower()
    if not lowered:
        return False
    # Typical virtual/non-hardware ALSA/PipeWire names.
    virtual_tokens = (
        "sysdefault",
        "default",
        "dmix",
        "dsnoop",
        "surround",
        "iec958",
        "pulse",
        "jack",
    )
    if any(token in lowered for token in virtual_tokens):
        return False
    return True


def _audio_runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    if not env.get("XDG_RUNTIME_DIR"):
        env["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
    return env


def _list_pipewire_sinks() -> tuple[list[dict[str, str]], str]:
    env = _audio_runtime_env()
    output = subprocess.check_output(
        ["pactl", "list", "short", "sinks"],
        text=True,
        env=env,
    )
    sinks: list[dict[str, str]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        sink_id = parts[0].strip()
        sink_name = parts[1].strip()
        state = parts[4].strip() if len(parts) > 4 else ""
        sinks.append({
            "id": sink_id,
            "name": sink_name,
            "label": sink_name,
            "state": state,
        })
    try:
        default_sink = subprocess.check_output(
            ["pactl", "get-default-sink"],
            text=True,
            env=env,
        ).strip()
    except Exception:
        default_sink = ""
    return sinks, default_sink


def _normalize_audio_preference(value: str) -> str:
    normalized = (value or "").strip().lower()
    # Hardware indices can change across reboots; ignore them for matching.
    normalized = re.sub(r"\s*\(hw:\d+,\d+\)\s*", " ", normalized)
    # Optional UI duplicate suffixes like "(#1)" should not affect matching.
    normalized = re.sub(r"\s*\(#\d+\)\s*", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _parse_hw_card_index(name: str) -> int | None:
    match = re.search(r"\(hw\s*:\s*(\d+)\s*,\s*\d+\)", name or "", re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _usb_path_for_card(card_index: int | None) -> str | None:
    if card_index is None:
        return None
    try:
        sysfs = f"/sys/class/sound/card{card_index}/device"
        real = os.path.realpath(sysfs)
        base = os.path.basename(real).split(":", 1)[0]
        if re.match(r"^\d+-\d+(?:\.\d+)*$", base):
            return base
        return None
    except Exception:
        return None


def _preference_from_device_name(device_name: str) -> str:
    card_index = _parse_hw_card_index(device_name)
    usb_path = _usb_path_for_card(card_index)
    if usb_path:
        return f"usbpath:{usb_path}"
    return str(device_name or "").strip()


def _preference_matches_device_name(preference: str, device_name: str) -> bool:
    preference = (preference or "").strip()
    if not preference:
        return True
    if preference.lower().startswith("usbpath:"):
        wanted = preference.split(":", 1)[1].strip().lower()
        candidate = _usb_path_for_card(_parse_hw_card_index(device_name))
        return bool(wanted and candidate and candidate.lower() == wanted)
    pref_norm = _normalize_audio_preference(preference)
    name_norm = _normalize_audio_preference(device_name)
    return pref_norm in name_norm or name_norm in pref_norm


def _resolve_selected_preference(
    saved_preference: str, devices: list[dict[str, str | int]]
) -> str:
    if not saved_preference:
        return ""

    for dev in devices:
        value = str(dev.get("value", ""))
        alias = str(dev.get("legacy_value", ""))
        if saved_preference in (value, alias):
            return value

    saved_norm = _normalize_audio_preference(saved_preference)
    for dev in devices:
        value = str(dev.get("value", ""))
        alias = str(dev.get("legacy_value", ""))
        if (
            _normalize_audio_preference(value) == saved_norm
            or _normalize_audio_preference(alias) == saved_norm
        ):
            return value

    return saved_preference


def get_usb_pcm_card_index(preference_override: str | None = None):
    raw_preference = (
        preference_override
        if preference_override is not None
        else core_config.SPEAKER_PREFERENCE
    )
    preference = (raw_preference or "").strip()
    normalized_preference = _normalize_audio_preference(preference)
    try:
        out = subprocess.check_output(["aplay", "-l"], text=True)
        cards = re.findall(
            r"card (\d+): ([^\s]+) \[(.*?)\], device (\d+): (.*?) \[", out
        )

        if preference.lower().startswith("usbpath:"):
            wanted_path = preference.split(":", 1)[1].strip().lower()
            for card_index, *_ in cards:
                if (_usb_path_for_card(int(card_index)) or "").lower() == wanted_path:
                    return int(card_index)

        # If SPEAKER_PREFERENCE includes "(hw:X,Y)", trust that card index first.
        if preference:
            hw_match = re.search(r"hw\s*:\s*(\d+)\s*,\s*\d+", preference.lower())
            if hw_match:
                card_from_hw = int(hw_match.group(1))
                if any(int(card_index) == card_from_hw for card_index, *_ in cards):
                    return card_from_hw

        if not preference:
            return None

        for card_index, shortname, longname, device_index, desc in cards:
            name = f"{shortname} {longname} {desc}"
            normalized_name = _normalize_audio_preference(name)
            if normalized_preference and (
                normalized_preference in normalized_name
                or normalized_name in normalized_preference
            ):
                return int(card_index)

        # Last fallback: first USB audio card from aplay list.
        for card_index, shortname, longname, _, desc in cards:
            combined = f"{shortname} {longname} {desc}".lower()
            if "usb" in combined:
                return int(card_index)

        return None
    except Exception as e:
        print("Failed to detect speaker card:", e)
        return None


def get_usb_capture_card_index():
    preference = (core_config.MIC_PREFERENCE or "").strip()
    normalized_pref = _normalize_audio_preference(preference)
    try:
        output = subprocess.check_output(["arecord", "-l"], text=True)
        cards = re.findall(
            r"card (\d+): ([^\s]+) \[(.*?)\], device (\d+): (.*?) \[", output
        )
        if preference.lower().startswith("usbpath:"):
            wanted_path = preference.split(":", 1)[1].strip().lower()
            for card_index, *_ in cards:
                if (_usb_path_for_card(int(card_index)) or "").lower() == wanted_path:
                    return int(card_index)
        for card_index, shortname, longname, device_index, desc in cards:
            name_combined = f"{shortname} {longname} {desc}"
            normalized_name = _normalize_audio_preference(name_combined)
            if normalized_pref and (
                normalized_pref in normalized_name or normalized_name in normalized_pref
            ):
                return int(card_index)
        for card_index, _, longname, _, _ in cards:
            if "usb" in longname.lower():
                return int(card_index)
        return None
    except Exception as e:
        print("Failed to detect mic card:", e)
        return None


def amixer_base_args_for_card(card_index):
    return ["-D", "default"] if card_index is None else ["-c", str(card_index)]


def alsa_play_device(card_index):
    return "default" if card_index is None else f"plughw:{card_index},0"


def get_mic_gain_numid(card_index):
    """Find the numid for mic gain on the specified card."""
    try:
        output = subprocess.check_output(
            ["amixer", "-c", str(card_index), "controls"], text=True
        )
        for line in output.splitlines():
            if "Mic Capture Volume" in line:
                match = re.search(r"numid=(\d+)", line)
                if match:
                    return int(match.group(1))
    except Exception as e:
        print("Failed to get mic gain numid:", e)
        return None


@bp.route("/audio/devices")
def audio_devices():
    try:
        devices = sd.query_devices()
        mic_preference = (core_config.MIC_PREFERENCE or "").strip()
        speaker_preference = (core_config.SPEAKER_PREFERENCE or "").strip()

        input_devices = []
        output_devices = []
        seen_inputs: set[str] = set()
        seen_outputs: set[str] = set()
        for idx, dev in enumerate(devices):
            name = str(dev.get("name", "")).strip()
            if not name:
                continue
            if not _is_selectable_audio_device_name(name):
                continue
            preference_value = _preference_from_device_name(name)
            card_index = _parse_hw_card_index(name)
            usb_path = _usb_path_for_card(card_index)
            label_suffix = f" (usb:{usb_path})" if usb_path else f" (#{idx})"
            if int(dev.get("max_input_channels", 0) or 0) > 0:
                key = preference_value.lower()
                if key in seen_inputs:
                    continue
                seen_inputs.add(key)
                input_devices.append({
                    "value": preference_value,
                    "legacy_value": name,
                    "label": f"{name}{label_suffix}",
                    "index": idx,
                })
            if int(dev.get("max_output_channels", 0) or 0) > 0:
                key = preference_value.lower()
                if key in seen_outputs:
                    continue
                seen_outputs.add(key)
                output_devices.append({
                    "value": preference_value,
                    "legacy_value": name,
                    "label": f"{name}{label_suffix}",
                    "index": idx,
                })

        selected_mic = _resolve_selected_preference(mic_preference, input_devices)
        selected_speaker = _resolve_selected_preference(
            speaker_preference, output_devices
        )

        return jsonify({
            "input_devices": input_devices,
            "output_devices": output_devices,
            "selected_mic": selected_mic,
            "selected_speaker": selected_speaker,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def audio_callback(indata, frames, time_info, status):
    if not mic_check_running:
        raise sd.CallbackStop()
    samples = indata.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(np.square(samples))))
    if np.issubdtype(indata.dtype, np.floating):
        max_abs = float(np.max(np.abs(samples))) if samples.size else 0.0
        if max_abs <= 1.5:
            rms *= 32768.0
    rms_queue.put(rms)


@bp.route("/wakeup", methods=["GET"])
def list_wakeup_clips():
    # Get current persona to check for persona-specific clips
    current_persona = "default"
    try:
        from core.persona_manager import persona_manager

        current_persona = persona_manager.current_persona
    except Exception:
        pass

    # Load wake-up data from the current persona's configuration
    config = configparser.ConfigParser()
    if current_persona == "default":
        # For default persona, use the main persona file
        config.read(PERSONA_PATH)
    else:
        # For other personas, use their specific persona file
        persona_file = os.path.join("personas", current_persona, "persona.ini")
        if os.path.exists(persona_file):
            config.read(persona_file)
        else:
            # Fallback to main persona file if persona file doesn't exist
            config.read(PERSONA_PATH)

    wakeup_data = dict(config["WAKEUP"]) if "WAKEUP" in config else {}

    # Check for appropriate wake-up files based on persona
    files = []
    if current_persona and current_persona != "default":
        # For non-default personas, check persona-specific directory
        persona_wakeup_dir = os.path.join("personas", current_persona, "wakeup")
        if os.path.exists(persona_wakeup_dir):
            files = glob.glob(os.path.join(persona_wakeup_dir, "*.wav"))
    elif current_persona == "default":
        # For default persona, use the custom directory
        files = glob.glob(str(WAKE_UP_DIR / "*.wav"))

    # If no files found yet, check custom directory as fallback
    if not files:
        files = glob.glob(str(WAKE_UP_DIR / "*.wav"))

    available = {os.path.splitext(os.path.basename(f))[0] for f in files}
    clips = []
    for k in sorted(wakeup_data.keys(), key=lambda x: int(x)):
        phrase = wakeup_data[k]
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", phrase).strip("_").lower()
        has_audio = slug in available or k in available
        clips.append({"index": int(k), "phrase": phrase, "has_audio": has_audio})
    return jsonify({"clips": clips})


@bp.route("/wakeup/play", methods=["POST"])
def play_wakeup_clip():
    try:
        index = int(request.json.get("index"))
        persona_name = request.json.get("persona")  # Get persona from request

        if index < 1 or index > 99:
            return jsonify({"error": "Invalid clip index"}), 400

        # If no persona specified, get current persona from persona manager
        if not persona_name:
            try:
                from core.persona_manager import persona_manager

                persona_name = persona_manager.current_persona
            except Exception:
                persona_name = "default"

        # Check appropriate directory based on persona
        sound_path = None
        if persona_name and persona_name != "default":
            # For non-default personas, check persona-specific directory
            persona_sound_path = os.path.join(
                PROJECT_ROOT, "personas", persona_name, "wakeup", f"{index}.wav"
            )
            if os.path.exists(persona_sound_path):
                sound_path = persona_sound_path
        elif persona_name == "default":
            # For default persona, use custom directory
            sound_path = os.path.join(
                PROJECT_ROOT, "sounds", "wake-up", "custom", f"{index}.wav"
            )

        # If no sound path found yet, check custom directory as fallback
        if not sound_path:
            sound_path = os.path.join(
                PROJECT_ROOT, "sounds", "wake-up", "custom", f"{index}.wav"
            )

        if not os.path.exists(sound_path):
            return jsonify({"error": f"Clip {index}.wav not found"}), 404

        # Prefer sending preview request to the running Billy service process.
        # That process owns GPIO, so mouth movement works reliably.
        mqtt_host = (core_config.MQTT_HOST or "").strip()
        mqtt_user = (core_config.MQTT_USERNAME or "").strip()
        mqtt_pass = (core_config.MQTT_PASSWORD or "").strip()
        mqtt_port = int(core_config.MQTT_PORT or 1883)
        if mqtt_host and mqtt_user and mqtt_pass:
            try:
                client = mqtt.Client()
                client.username_pw_set(mqtt_user, mqtt_pass)
                client.connect(mqtt_host, mqtt_port, 5)
                payload = json.dumps({
                    "index": index,
                    "persona": persona_name,
                })
                result = client.publish("billy/wakeup/play", payload, retain=False)
                client.disconnect()
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    return jsonify({
                        "status": f"Queued clip {index}.wav via MQTT for Billy service playback"
                    })
            except Exception:
                pass

        # Fallback: plain local playback without touching Billy GPIO ownership.
        card_index = get_usb_pcm_card_index()
        device = alsa_play_device(card_index)
        subprocess.Popen(["aplay", "-q", "-D", device, sound_path])
        return jsonify({"status": f"Playing clip {index}.wav on {device}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/sounds/wake-up/<filename>")
def serve_wakeup_sound(filename):
    return send_from_directory("sounds/wake-up", filename)


@bp.route("/wakeup/generate", methods=["POST"])
def generate_wakeup_clip():
    data = request.get_json()
    prompt = data.get("text", "").strip()
    index = data.get("index")
    persona_name = data.get("persona")  # Get persona from request

    if not prompt or index is None:
        return jsonify({"error": "Missing 'text' or 'index'"}), 400

    # If no persona specified, get current persona from persona manager
    if not persona_name:
        try:
            from core.persona_manager import persona_manager

            persona_name = persona_manager.current_persona
        except Exception:
            persona_name = "default"

    try:
        path = generate_wake_clip_async(prompt, index, persona_name)
        return jsonify({"status": "ok", "path": path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/wakeup/remove", methods=["POST"])
def remove_wakeup_clip():
    data = request.get_json()
    index_to_remove = str(data.get("index"))

    # Get current persona to determine which file to modify
    current_persona = "default"
    try:
        from core.persona_manager import persona_manager

        current_persona = persona_manager.current_persona
    except Exception:
        pass

    # Determine the file path based on current persona
    if current_persona == "default":
        persona_file = PERSONA_PATH
    else:
        from pathlib import Path

        personas_dir = Path("personas")
        persona_file = personas_dir / current_persona / "persona.ini"

    config = configparser.ConfigParser()
    config.read(persona_file)
    if "WAKEUP" not in config:
        return jsonify({"error": "No wakeup section found"}), 400
    wakeup = dict(config["WAKEUP"])
    if index_to_remove not in wakeup:
        return jsonify({"error": f"Clip {index_to_remove} not found"}), 404
    removed_phrase = wakeup.pop(index_to_remove)
    new_wakeup = {}
    old_to_new_index = {}
    for i, (old_k, phrase) in enumerate(wakeup.items(), start=1):
        new_wakeup[str(i)] = phrase
        old_to_new_index[old_k] = str(i)
    config["WAKEUP"] = new_wakeup
    with open(persona_file, "w") as f:
        config.write(f)
    audio_path_num = WAKE_UP_DIR / f"{index_to_remove}.wav"
    audio_path_slug = (
        WAKE_UP_DIR
        / f"{re.sub(r'[^a-zA-Z0-9_-]+', '_', removed_phrase).strip('_').lower()}.wav"
    )
    for p in (audio_path_num, audio_path_slug):
        if p.exists():
            p.unlink()
    for old_k, new_k in old_to_new_index.items():
        old_path = WAKE_UP_DIR / f"{old_k}.wav"
        new_path = WAKE_UP_DIR / f"{new_k}.wav"
        if old_path.exists() and old_path != new_path:
            old_path.rename(new_path)
    return jsonify({"status": "removed and reindexed"})


@bp.route("/speaker-test", methods=["POST"])
def speaker_test():
    try:
        data = request.get_json(silent=True) or {}
        speaker_preference = str(data.get("speaker_preference") or "").strip() or None
        sound_path = os.path.join(PROJECT_ROOT, "sounds", "speakertest.wav")
        card_index = get_usb_pcm_card_index(speaker_preference)
        device = alsa_play_device(card_index)
        subprocess.Popen(["aplay", "-q", "-D", device, sound_path])
        return jsonify({"status": f"playing on {device}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/mic-check")
def mic_check():
    def rms_stream_generator():
        # use module-level flag
        global mic_check_running
        mic_check_running = True
        try:
            from core import audio as core_audio

            with sd.InputStream(
                samplerate=core_audio.MIC_RATE,
                device=core_audio.MIC_DEVICE_INDEX,
                channels=core_audio.MIC_CHANNELS,
                dtype="int16",
                blocksize=core_audio.CHUNK_SIZE,
                callback=audio_callback,
            ):
                while mic_check_running:
                    try:
                        rms = rms_queue.get(timeout=1.0)
                        payload = {
                            "rms": round(rms, 4),
                            "threshold": round(float(core_config.SILENCE_THRESHOLD), 4),
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                    except queue.Empty:
                        continue
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(rms_stream_generator(), mimetype="text/event-stream")


@bp.route("/mic-check/stop")
def mic_check_stop():
    global mic_check_running
    mic_check_running = False
    return jsonify({"status": "stopped"})


@bp.route("/mic-gain", methods=["GET", "POST"])
def mic_gain():
    card_index = get_usb_capture_card_index()
    numid = get_mic_gain_numid(card_index)
    if card_index is None or numid is None:
        return jsonify({"error": "Could not determine mic card or control ID"}), 500
    if request.method == "GET":
        try:
            output = subprocess.check_output(
                ["amixer", "-c", str(card_index), "cget", f"numid={numid}"], text=True
            )
            match = re.search(r": values=(\d+)", output)
            gain = int(match.group(1)) if match else None
            return jsonify({"gain": gain})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    if request.method == "POST":
        try:
            data = request.get_json()
            value = int(data.get("value", 8))
            if 0 <= value <= 16:
                subprocess.check_call([
                    "amixer",
                    "-c",
                    str(card_index),
                    "cset",
                    f"numid={numid}",
                    str(value),
                ])
                return "OK"
            return jsonify({"error": "Mic gain must be between 0 and 16"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Unsupported method"}), 405


@bp.route("/volume", methods=["GET", "POST"])
def volume():
    try:
        card_index = get_usb_pcm_card_index()
        base = amixer_base_args_for_card(card_index)
        control = "PCM"
        if request.method == "GET":
            output = subprocess.check_output(
                ["amixer", *base, "get", control], text=True
            )
            m = re.search(r"\[(\d{1,3})%\]", output)
            if not m:
                return jsonify({"error": f"Could not parse volume for {control}"}), 500
            return jsonify({
                "volume": int(m.group(1)),
                "control": control,
                "target": "default" if card_index is None else f"card {card_index}",
            })
        data = request.get_json()
        if data is None or "volume" not in data:
            return jsonify({"error": "Missing volume"}), 400
        value = int(data["volume"])
        if not (0 <= value <= 100):
            return jsonify({"error": "Volume must be 0–100"}), 400
        subprocess.check_call(["amixer", *base, "set", control, f"{value}%"])
        return jsonify({
            "volume": value,
            "control": control,
            "target": "default" if card_index is None else f"card {card_index}",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/device-info")
def device_info():
    try:
        max_attempts = 6
        for attempt in range(max_attempts):
            mic_name = "Unknown"
            speaker_name = "Unknown"
            fallback_mic = None
            fallback_speaker = None
            devices = sd.query_devices()
            for dev in devices:
                if fallback_mic is None and dev["max_input_channels"] > 0:
                    fallback_mic = dev["name"]
                if fallback_speaker is None and dev["max_output_channels"] > 0:
                    fallback_speaker = dev["name"]
                if (
                    mic_name == "Unknown"
                    and dev["max_input_channels"] > 0
                    and _preference_matches_device_name(
                        core_config.MIC_PREFERENCE, str(dev["name"])
                    )
                ):
                    mic_name = dev["name"]
                if (
                    speaker_name == "Unknown"
                    and dev["max_output_channels"] > 0
                    and _preference_matches_device_name(
                        core_config.SPEAKER_PREFERENCE, str(dev["name"])
                    )
                ):
                    speaker_name = dev["name"]
            # If preferred device is not matchable yet, still show first available
            # device so UI does not remain "Unknown" while audio itself is usable.
            if mic_name == "Unknown" and fallback_mic:
                mic_name = fallback_mic
            if speaker_name == "Unknown" and fallback_speaker:
                speaker_name = fallback_speaker
            if mic_name != "Unknown" or speaker_name != "Unknown":
                return jsonify({"mic": mic_name, "speaker": speaker_name})
            if attempt < max_attempts - 1:
                time.sleep(0.4)
        return jsonify({"mic": mic_name, "speaker": speaker_name})
    except Exception as e:
        return jsonify({"mic": "Unknown", "speaker": "Unknown", "error": str(e)}), 500
