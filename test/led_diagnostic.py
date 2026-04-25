#!/usr/bin/env python3
"""Simple LED diagnostic for GPIO pin tests."""

import argparse
import importlib
import sys
import time
import types
from pathlib import Path
from subprocess import DEVNULL, run


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn an LED on/off or flash it to verify GPIO wiring."
    )
    parser.add_argument(
        "--pin",
        type=positive_int,
        default=25,
        help="GPIO pin number used for the LED (default: 25).",
    )
    parser.add_argument(
        "--action",
        choices=("all", "on", "off", "flash"),
        default="all",
        help="Action to run (default: all).",
    )
    parser.add_argument(
        "--flash-count",
        type=positive_int,
        default=3,
        help="Number of flashes when using --action flash/all.",
    )
    parser.add_argument(
        "--on-time",
        type=positive_float,
        default=0.3,
        help="Seconds LED stays on during flash.",
    )
    parser.add_argument(
        "--off-time",
        type=positive_float,
        default=0.3,
        help="Seconds LED stays off during flash.",
    )
    parser.add_argument(
        "--hold",
        type=non_negative_float,
        default=1.0,
        help="Seconds to hold steady on/off states in --action all mode.",
    )
    parser.add_argument(
        "--skip-service-check",
        action="store_true",
        help="Run even if billy.service appears active.",
    )
    return parser.parse_args()


def billy_service_is_active() -> bool:
    try:
        result = run(
            ["systemctl", "is-active", "--quiet", "billy.service"],
            stdout=DEVNULL,
            stderr=DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0


def prepare_core_package() -> None:
    """Load core submodules without running core/__init__.py side effects."""
    if "core" in sys.modules:
        return

    core_package = types.ModuleType("core")
    core_package.__file__ = str(REPO_ROOT / "core" / "__init__.py")
    core_package.__path__ = [str(REPO_ROOT / "core")]
    sys.modules["core"] = core_package


def main() -> int:
    args = parse_args()

    if billy_service_is_active() and not args.skip_service_check:
        print(
            "billy.service is active. Stop it first, or rerun with "
            "--skip-service-check if GPIO is free."
        )
        return 1

    prepare_core_package()
    led_module = importlib.import_module("core.led")
    LEDController = led_module.LEDController
    lgpio_available = bool(getattr(led_module, "lgpio_available", False))
    mockfish_enabled = bool(getattr(led_module, "MOCKFISH", False))

    if not lgpio_available and not mockfish_enabled:
        print(
            "lgpio is not installed in this Python environment. "
            "The LED diagnostic will not control real GPIO.\n"
            "Activate your venv and install requirements:\n"
            "  source venv/bin/activate\n"
            "  pip3 install -r requirements.txt"
        )
        return 2

    try:
        with LEDController(pin=args.pin) as led:
            if args.action == "on":
                print(f"LED ON (GPIO {args.pin})")
                led.on()
                if args.hold > 0:
                    time.sleep(args.hold)

            if args.action == "off":
                print(f"LED OFF (GPIO {args.pin})")
                led.off()
                if args.hold > 0:
                    time.sleep(args.hold)

            if args.action == "all":
                print(f"LED ON (GPIO {args.pin})")
                led.on()
                if args.hold > 0:
                    time.sleep(args.hold)

                print(f"LED OFF (GPIO {args.pin})")
                led.off()
                if args.hold > 0:
                    time.sleep(args.hold)

            if args.action in ("all", "flash"):
                print(
                    f"LED FLASH x{args.flash_count} "
                    f"(on={args.on_time}s off={args.off_time}s)"
                )
                led.flash(
                    times=args.flash_count, on_time=args.on_time, off_time=args.off_time
                )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

    print("LED diagnostic complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
