import atexit
import os
import time
from threading import Lock
from threading import Event, Thread

from .logger import logger


def _is_mockfish_enabled() -> bool:
    return os.getenv("MOCKFISH", "false").strip().lower() == "true"


MOCKFISH = _is_mockfish_enabled()


try:
    import lgpio

    lgpio_available = True
except ImportError:
    lgpio_available = False


if MOCKFISH or not lgpio_available:
    class MockLgpio:
        error = Exception

        @staticmethod
        def gpiochip_open(chip):
            return "mock_handle"

        @staticmethod
        def gpio_claim_output(handle, pin):
            pass

        @staticmethod
        def gpio_write(handle, pin, value):
            pass

        @staticmethod
        def gpio_free(handle, pin):
            pass

        @staticmethod
        def gpiochip_close(handle):
            pass

    lgpio = MockLgpio
    if MOCKFISH:
        logger.info("Mockfish: LED GPIO mocked for development", "🐟")
    elif not lgpio_available:
        logger.info("lgpio not available: LED GPIO mocked", "🐟")


class LEDController:
    """Simple controller for a single GPIO LED."""

    def __init__(self, pin: int = 25, chip: int = 0):
        self.pin = pin
        self.chip = chip
        self._lock = Lock()
        self._handle = lgpio.gpiochip_open(chip)
        self._active = True
        self._is_on = False

        lgpio.gpio_claim_output(self._handle, self.pin)
        lgpio.gpio_write(self._handle, self.pin, 0)
        logger.info(f"LED initialized on GPIO {self.pin}", "💡")

    @property
    def is_on(self) -> bool:
        return self._is_on

    def on(self):
        with self._lock:
            if not self._active:
                return
            lgpio.gpio_write(self._handle, self.pin, 1)
            self._is_on = True

    def off(self):
        with self._lock:
            if not self._active:
                return
            lgpio.gpio_write(self._handle, self.pin, 0)
            self._is_on = False

    def flash(self, times: int = 3, on_time: float = 0.25, off_time: float = 0.25):
        if times < 1:
            return
        for _ in range(times):
            self.on()
            time.sleep(on_time)
            self.off()
            time.sleep(off_time)

    def cleanup(self):
        with self._lock:
            if not self._active:
                return
            try:
                lgpio.gpio_write(self._handle, self.pin, 0)
                lgpio.gpio_free(self._handle, self.pin)
                lgpio.gpiochip_close(self._handle)
            except Exception as e:
                logger.warning(f"LED cleanup error on GPIO {self.pin}: {e}", "⚠️")
            finally:
                self._active = False
                self._is_on = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def _is_led_enabled() -> bool:
    value = os.getenv("LED_ENABLED", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _get_led_pin() -> int:
    raw = os.getenv("LED_PIN", "25").strip()
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"Invalid LED_PIN '{raw}', falling back to 25", "⚠️")
        return 25


def _get_thinking_blink_seconds() -> float:
    raw = os.getenv("LED_THINKING_BLINK_SECONDS", "0.2").strip()
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            f"Invalid LED_THINKING_BLINK_SECONDS '{raw}', falling back to 0.2",
            "⚠️",
        )
        return 0.2
    return max(0.05, value)


class LEDIndicator:
    """Session-level LED state manager: idle, listening, thinking."""

    VALID_MODES = {"idle", "listening", "thinking"}

    def __init__(self, enabled: bool | None = None, pin: int | None = None):
        self.enabled = _is_led_enabled() if enabled is None else enabled
        self.pin = _get_led_pin() if pin is None else pin
        self._lock = Lock()
        self._mode = "idle"
        self._blink_thread: Thread | None = None
        self._blink_stop = Event()
        self._blink_seconds = _get_thinking_blink_seconds()
        self._controller: LEDController | None = None

        if self.enabled:
            try:
                self._controller = LEDController(pin=self.pin)
                logger.info(
                    f"LED indicator enabled (pin={self.pin}, blink={self._blink_seconds}s)",
                    "💡",
                )
            except Exception as e:
                self.enabled = False
                self._controller = None
                logger.warning(f"LED indicator disabled after init error: {e}", "⚠️")
        else:
            logger.verbose("LED indicator disabled (LED_ENABLED=false)", "💡")

    def get_mode(self) -> str:
        with self._lock:
            return self._mode

    def set_mode(self, mode: str):
        normalized = (mode or "idle").strip().lower()
        if normalized not in self.VALID_MODES:
            logger.warning(f"Unknown LED mode '{mode}', ignoring", "⚠️")
            return

        with self._lock:
            previous = self._mode
            if previous == normalized and normalized != "thinking":
                return

            self._stop_blink_locked()
            self._mode = normalized

            if not self._controller:
                return

            if normalized == "idle":
                self._controller.off()
            elif normalized == "listening":
                self._controller.on()
            else:
                self._start_thinking_blink_locked()

    def _start_thinking_blink_locked(self):
        if not self._controller:
            return

        self._blink_stop.clear()

        def _worker():
            assert self._controller is not None
            led_on = False
            while not self._blink_stop.is_set():
                if led_on:
                    self._controller.off()
                else:
                    self._controller.on()
                led_on = not led_on
                if self._blink_stop.wait(self._blink_seconds):
                    break
            self._controller.off()

        self._blink_thread = Thread(target=_worker, daemon=True)
        self._blink_thread.start()

    def _stop_blink_locked(self):
        if self._blink_thread and self._blink_thread.is_alive():
            self._blink_stop.set()
            self._blink_thread.join(timeout=1.0)
        self._blink_thread = None
        self._blink_stop.clear()

    def cleanup(self):
        with self._lock:
            self._stop_blink_locked()
            if self._controller:
                self._controller.cleanup()
            self._controller = None
            self._mode = "idle"


_indicator: LEDIndicator | None = None
_indicator_lock = Lock()


def get_led_indicator() -> LEDIndicator:
    global _indicator
    with _indicator_lock:
        if _indicator is None:
            _indicator = LEDIndicator()
        return _indicator


def set_led_mode(mode: str):
    get_led_indicator().set_mode(mode)


def get_led_mode() -> str:
    return get_led_indicator().get_mode()


class temporary_led_mode:
    """Context manager to switch LED mode briefly and restore previous mode."""

    def __init__(self, mode: str):
        self._mode = mode
        self._previous = "idle"

    def __enter__(self):
        indicator = get_led_indicator()
        self._previous = indicator.get_mode()
        indicator.set_mode(self._mode)
        return indicator

    def __exit__(self, exc_type, exc_val, exc_tb):
        indicator = get_led_indicator()
        # Only restore if no other subsystem changed the mode while active.
        if indicator.get_mode() == self._mode:
            indicator.set_mode(self._previous)


def cleanup_led():
    global _indicator
    with _indicator_lock:
        if _indicator is None:
            return
        _indicator.cleanup()
        _indicator = None


atexit.register(cleanup_led)
