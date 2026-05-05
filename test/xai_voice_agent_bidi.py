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

import sounddevice as sd
import websockets.asyncio.client


SAMPLE_RATE = 24000
CHANNELS = 1
FRAME_MS = 40
FRAME_COUNT = int(SAMPLE_RATE * FRAME_MS / 1000)
DEFAULT_MODEL = "grok-voice-think-fast-1.0"
DEFAULT_VOICE = "rex"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="xAI bidirectional voice-agent test")
    parser.add_argument("--api-key", default="", help="xAI API key (optional)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Voice model")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Voice name")
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
                "input": {"format": {"type": "audio/pcm", "rate": SAMPLE_RATE}},
                "output": {"format": {"type": "audio/pcm", "rate": SAMPLE_RATE}},
            },
        },
    }


async def run_bidi_session(args: argparse.Namespace) -> int:
    api_key = (args.api_key or os.getenv("XAI_API_KEY", "")).strip()
    if not api_key:
        print("Missing API key. Set XAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

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
                    speaker_stream.write(base64.b64decode(delta))
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
            print("Connected. Sending session.update...")
            await ws.send(json.dumps(_build_session_update(args.voice, args.instructions)))
            print("Live. Speak into the mic. Press Ctrl+C to stop.")

            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=FRAME_COUNT,
                dtype="int16",
                channels=CHANNELS,
                callback=mic_callback,
            ), sd.RawOutputStream(
                samplerate=SAMPLE_RATE,
                dtype="int16",
                channels=CHANNELS,
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
