#!/usr/bin/env python3
"""
Minimal bidirectional audio test for xAI Voice Agent websocket.

Mic audio is streamed to xAI, and audio deltas from xAI are played on speakers.

Usage:
  XAI_API_KEY=... python test/xai_voice_agent_bidi.py
  python test/xai_voice_agent_bidi.py --api-key xai-... --model grok-voice-think-fast-1.0
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from typing import Any

import numpy as np
import sounddevice as sd
import websockets.asyncio.client

try:
    from scipy.signal import resample
except Exception:  # pragma: no cover - fallback when scipy is unavailable
    resample = None


API_SAMPLE_RATE = 24000
CHANNELS = 1
FRAME_MS = 40
DEFAULT_MODEL = "grok-voice-think-fast-1.0"
DEFAULT_VOICE = "rex"
PREFERRED_SAMPLE_RATES = (24000, 48000, 44100, 32000, 22050, 16000, 8000)


def _resample_pcm16_mono(chunk: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate or not chunk:
        return chunk

    samples = np.frombuffer(chunk, dtype=np.int16)
    if samples.size == 0:
        return b""

    target_len = max(1, int(round(samples.size * dst_rate / src_rate)))
    samples_f32 = samples.astype(np.float32)
    if resample is not None:
        out = resample(samples_f32, target_len)
    else:
        # Fallback linear resample if scipy is not available in the environment.
        src_x = np.arange(samples_f32.size, dtype=np.float32)
        dst_x = np.linspace(0.0, samples_f32.size - 1, target_len, dtype=np.float32)
        out = np.interp(dst_x, src_x, samples_f32)

    out_i16 = np.clip(np.rint(out), -32768, 32767).astype(np.int16)
    return out_i16.tobytes()


def _parse_device(value: str) -> int | str | None:
    text = (value or "").strip()
    if not text or text.lower() == "default":
        return None
    if text.isdigit():
        return int(text)
    return text


def _default_samplerate(kind: str, device: int | str | None) -> int | None:
    try:
        info = sd.query_devices(device=device, kind=kind)
    except Exception:
        return None
    rate = info.get("default_samplerate")
    if not rate:
        return None
    return int(round(float(rate)))


def _pick_supported_samplerate(
    *,
    kind: str,
    device: int | str | None,
    requested_rate: int,
) -> int:
    checker = sd.check_input_settings if kind == "input" else sd.check_output_settings

    if requested_rate > 0:
        checker(
            device=device,
            channels=CHANNELS,
            dtype="int16",
            samplerate=requested_rate,
        )
        return requested_rate

    candidates: list[int] = []
    default_rate = _default_samplerate(kind, device)
    if default_rate:
        candidates.append(default_rate)
    for rate in PREFERRED_SAMPLE_RATES:
        if rate not in candidates:
            candidates.append(rate)

    for rate in candidates:
        try:
            checker(device=device, channels=CHANNELS, dtype="int16", samplerate=rate)
            return rate
        except Exception:
            continue

    raise RuntimeError(
        f"No supported {kind} sample rate found for device={device!r}. "
        "Run with --list-devices and then set --input-device/--output-device and rates."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="xAI bidirectional voice-agent test")
    parser.add_argument("--api-key", default="", help="xAI API key (optional)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Voice model")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Voice name")
    parser.add_argument(
        "--input-device",
        default="",
        help="Input device id or substring (default: system default)",
    )
    parser.add_argument(
        "--output-device",
        default="",
        help="Output device id or substring (default: system default)",
    )
    parser.add_argument(
        "--input-rate",
        type=int,
        default=0,
        help="Input sample rate override (0 = auto)",
    )
    parser.add_argument(
        "--output-rate",
        type=int,
        default=0,
        help="Output sample rate override (0 = auto)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available sound devices and exit",
    )
    parser.add_argument(
        "--instructions",
        default="You are a helpful assistant. Keep answers short.",
        help="Session instructions",
    )
    parser.add_argument("--ping-interval", type=float, default=20.0)
    parser.add_argument("--ping-timeout", type=float, default=60.0)
    return parser.parse_args()


def _build_session_update(voice: str, instructions: str) -> dict[str, Any]:
    return {
        "type": "session.update",
        "session": {
            "voice": voice,
            "instructions": instructions,
            "turn_detection": {
                "type": "server_vad",
                "create_response": True,
                "interrupt_response": True,
            },
            "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": API_SAMPLE_RATE}},
                "output": {"format": {"type": "audio/pcm", "rate": API_SAMPLE_RATE}},
            },
        },
    }


async def run_bidi_session(args: argparse.Namespace) -> int:
    if args.list_devices:
        print(sd.query_devices())
        return 0

    api_key = (args.api_key or os.getenv("XAI_API_KEY", "")).strip()
    if not api_key:
        print("Missing API key. Set XAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    input_device = _parse_device(args.input_device)
    output_device = _parse_device(args.output_device)
    try:
        input_rate = _pick_supported_samplerate(
            kind="input",
            device=input_device,
            requested_rate=args.input_rate,
        )
        output_rate = _pick_supported_samplerate(
            kind="output",
            device=output_device,
            requested_rate=args.output_rate,
        )
    except Exception as exc:
        print(f"Audio device configuration failed: {exc}")
        return 1

    frame_count = max(1, int(input_rate * FRAME_MS / 1000))

    uri = f"wss://api.x.ai/v1/realtime?model={args.model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=64)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_stop() -> None:
        if not stop_event.is_set():
            stop_event.set()

    def mic_callback(indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            print(f"[mic] {status}", file=sys.stderr)

        chunk = bytes(indata)
        if input_rate != API_SAMPLE_RATE:
            chunk = _resample_pcm16_mono(chunk, input_rate, API_SAMPLE_RATE)
        if not chunk:
            return

        def _enqueue() -> None:
            if audio_queue.full():
                try:
                    audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                audio_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                pass

        loop.call_soon_threadsafe(_enqueue)

    async def sender(ws) -> None:
        while not stop_event.is_set():
            try:
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            payload = {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("ascii"),
            }
            await ws.send(json.dumps(payload))

    async def receiver(ws, speaker_stream: sd.RawOutputStream) -> None:
        async for message in ws:
            event = json.loads(message)
            event_type = str(event.get("type", ""))

            if event_type == "response.output_audio.delta":
                delta = event.get("delta")
                if isinstance(delta, str) and delta:
                    pcm = base64.b64decode(delta)
                    if output_rate != API_SAMPLE_RATE:
                        pcm = _resample_pcm16_mono(pcm, API_SAMPLE_RATE, output_rate)
                    if pcm:
                        speaker_stream.write(pcm)
            elif event_type in {"response.output_text.delta", "response.text.delta"}:
                delta = event.get("delta")
                if isinstance(delta, str) and delta:
                    print(delta, end="", flush=True)
            elif event_type == "response.done":
                print()
            elif event_type == "error":
                print("\n[server error]")
                print(json.dumps(event, indent=2))
                request_stop()
                break

            if stop_event.is_set():
                break

    print(f"Connecting: {uri}")
    try:
        async with websockets.asyncio.client.connect(
            uri,
            additional_headers=headers,
            open_timeout=30,
            ping_interval=args.ping_interval,
            ping_timeout=args.ping_timeout,
            close_timeout=2,
        ) as ws:
            print(
                "Audio config: "
                f"input_device={input_device!r} input_rate={input_rate} "
                f"output_device={output_device!r} output_rate={output_rate} "
                f"api_rate={API_SAMPLE_RATE}"
            )
            print("Connected. Sending session.update...")
            await ws.send(json.dumps(_build_session_update(args.voice, args.instructions)))
            print("Live. Speak into the mic. Press Ctrl+C to stop.")

            with sd.RawInputStream(
                samplerate=input_rate,
                blocksize=frame_count,
                dtype="int16",
                channels=CHANNELS,
                device=input_device,
                callback=mic_callback,
            ), sd.RawOutputStream(
                samplerate=output_rate,
                dtype="int16",
                channels=CHANNELS,
                device=output_device,
            ) as speaker_stream:
                send_task = asyncio.create_task(sender(ws))
                recv_task = asyncio.create_task(receiver(ws, speaker_stream))
                stop_task = asyncio.create_task(stop_event.wait())

                done, pending = await asyncio.wait(
                    {send_task, recv_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

            return 0
    except websockets.exceptions.InvalidStatus as exc:
        print(f"WebSocket rejected during handshake: HTTP {exc.response.status_code}")
        return 1
    except Exception as exc:
        print(f"Voice agent test failed: {type(exc).__name__}: {exc}")
        return 1


def main() -> int:
    args = parse_args()

    try:
        return asyncio.run(run_bidi_session(args))
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
