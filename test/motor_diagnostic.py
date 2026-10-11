#!/usr/bin/env python3
"""Cycle Billy's motors to help isolate wiring or hardware faults."""

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


def percentage(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 100:
        raise argparse.ArgumentTypeError("value must be between 1 and 100")
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
        description=(
            "Pulse Billy's head, tail, and mouth motors so you can verify "
            "movement and spot wiring issues."
        )
    )
    parser.add_argument(
        "--motor",
        choices=("all", "head", "tail", "mouth"),
        default="all",
        help="Which motor to test. Defaults to all motors in sequence.",
    )
    parser.add_argument(
        "--cycles",
        type=positive_int,
        default=3,
        help="How many times to cycle each selected motor.",
    )
    parser.add_argument(
        "--head-hold",
        type=positive_float,
        default=1.0,
        help="Seconds to leave the head motor engaged before retracting.",
    )
    parser.add_argument(
        "--head-rest",
        type=non_negative_float,
        default=0.6,
        help="Seconds to wait after retracting the head between cycles.",
    )
    parser.add_argument(
        "--tail-duration",
        type=positive_float,
        default=0.25,
        help="Seconds to drive the tail motor for each pulse.",
    )
    parser.add_argument(
        "--tail-rest",
        type=non_negative_float,
        default=0.35,
        help="Seconds to wait between tail pulses.",
    )
    parser.add_argument(
        "--mouth-duration",
        type=positive_float,
        default=0.2,
        help="Seconds to drive the mouth motor for each pulse.",
    )
    parser.add_argument(
        "--mouth-rest",
        type=non_negative_float,
        default=0.3,
        help="Seconds to wait between mouth pulses.",
    )
    parser.add_argument(
        "--mouth-speed",
        type=percentage,
        default=100,
        help="PWM duty percentage for the mouth motor.",
    )
    parser.add_argument(
        "--skip-service-check",
        action="store_true",
        help="Run even if billy.service appears to be active.",
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


def selected_motors(name: str) -> list[str]:
    if name == "all":
        return ["head", "tail", "mouth"]
    return [name]


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
            "billy.service is active. Stop it before running this diagnostic, "
            "or rerun with --skip-service-check if you know GPIO is free."
        )
        return 1

    prepare_core_package()
    config = importlib.import_module("core.config")
    movements = importlib.import_module("core.movements")

    BILLY_MODEL = config.BILLY_MODEL
    BILLY_PINS = config.BILLY_PINS
    cleanup_gpio = movements.cleanup_gpio
    move_head = movements.move_head
    move_mouth = movements.move_mouth
    move_tail = movements.move_tail
    stop_all_motors = movements.stop_all_motors
    stop_mouth = movements.stop_mouth

    head_hold = max(args.head_hold, 0.7)
    if head_hold != args.head_hold:
        print("Adjusted --head-hold to 0.7s so the extend cycle can finish cleanly.")

    def cycle_head() -> None:
        move_head("off")
        time.sleep(0.2)
        for index in range(1, args.cycles + 1):
            print(f"Head cycle {index}/{args.cycles}")
            move_head("on")
            time.sleep(head_hold)
            move_head("off")
            time.sleep(args.head_rest)

    def cycle_tail() -> None:
        for index in range(1, args.cycles + 1):
            print(f"Tail cycle {index}/{args.cycles}")
            move_tail(duration=args.tail_duration)
            time.sleep(args.tail_duration + args.tail_rest)

    def cycle_mouth() -> None:
        for index in range(1, args.cycles + 1):
            print(f"Mouth cycle {index}/{args.cycles}")
            move_mouth(args.mouth_speed, args.mouth_duration, brake=True)
            time.sleep(args.mouth_duration + args.mouth_rest)
        stop_mouth()

    runners = {
        "head": cycle_head,
        "tail": cycle_tail,
        "mouth": cycle_mouth,
    }

    print(
        f"Starting motor diagnostic for {', '.join(selected_motors(args.motor))} "
        f"(model={BILLY_MODEL}, pins={BILLY_PINS}, cycles={args.cycles})."
    )

    try:
        for motor in selected_motors(args.motor):
            print(f"\nTesting {motor} motor...")
            runners[motor]()
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nInterrupted. Stopping all motors.")
        return 130
    finally:
        stop_all_motors()
        cleanup_gpio()

    print("\nMotor diagnostic complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
