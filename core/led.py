import os
import time
from threading import Lock

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
