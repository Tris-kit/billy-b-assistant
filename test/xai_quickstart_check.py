#!/usr/bin/env python3
"""
Minimal xAI Quickstart test using the Responses API.

Usage:
  XAI_API_KEY=... python test/xai_quickstart_check.py
  python test/xai_quickstart_check.py --api-key xai-... --model grok-4.3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests


API_URL = "https://api.x.ai/v1/responses"
DEFAULT_MODEL = "grok-4.3"


def _extract_text(response_json: dict) -> str:
    direct = response_json.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    outputs = response_json.get("output")
    if not isinstance(outputs, list):
        return ""

    text_chunks: list[str] = []
    for item in outputs:
        content = item.get("content") if isinstance(item, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            value = part.get("text")
            if isinstance(value, str) and value.strip():
                text_chunks.append(value.strip())
    return "\n".join(text_chunks).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="xAI quickstart connectivity test")
    parser.add_argument("--api-key", default="", help="xAI API key (optional)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: xai connectivity ok",
        help="User prompt",
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = (args.api_key or os.getenv("XAI_API_KEY", "")).strip()
    if not api_key:
        print("Missing API key. Set XAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    payload = {
        "model": args.model,
        "input": [
            {
                "role": "system",
                "content": "You are a concise assistant used for API diagnostics.",
            },
            {"role": "user", "content": args.prompt},
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=args.timeout,
        )
    except requests.RequestException as exc:
        print(f"Request failed before HTTP response: {exc}", file=sys.stderr)
        return 1

    request_id = response.headers.get("x-request-id", "")
    print(f"HTTP {response.status_code} from {API_URL}")
    if request_id:
        print(f"x-request-id: {request_id}")

    try:
        body = response.json()
    except ValueError:
        print("Non-JSON response body:")
        print(response.text[:2000])
        return 1

    if response.ok:
        text = _extract_text(body)
        print("Response text:")
        print(text or "<no output_text field>")
        return 0

    print("Error JSON:")
    print(json.dumps(body, indent=2)[:4000])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
