"""State machine for Billy session turn management."""

import threading
import time
from typing import Any

from ..config import HEAD_RETRACT_DELAY_SECONDS, SILENCE_THRESHOLD
from ..led import set_led_mode
from ..logger import logger
from ..movements import move_head
from ..mqtt import mqtt_publish


class SessionState:
    """Manages session state and turn detection."""

    def __init__(self, session):
        self.session = session
        self.full_response_text = ""
        self.allow_mic_input = True
        self.assistant_speaking = False
        self.response_active = False

        # Turn-level flags
        self._turn_announced = False
        self._saw_transcript_delta = False
        self._turn_had_speech = False
        self._active_transcript_stream: str | None = None
        self._added_done_text = False
        self._saw_follow_up_call = False
        self._triggered_new_response = False
        self.follow_up_retry_count = 0
        self._skip_post_response_once = False
        self._last_heuristic_signature: tuple[str, bool] | None = None
        self._last_user_turn_meaningful = False

        # Follow-up detection
        self.follow_up_expected = False
        self.follow_up_prompt: str | None = None

        # Short audio turn detection
        self._ignore_next_short_audio_response = False
        self._pending_input_audio_chunks = 0
        self._last_committed_audio_chunks = 0
        self._pending_loud_audio_chunks = 0
        self._last_committed_loud_audio_chunks = 0
        self._pending_peak_rms = 0.0
        self._last_committed_peak_rms = 0.0
        self._current_input_had_server_speech = False
        self._last_committed_had_server_speech = False
        self._server_input_speaking = False
        self._head_retract_timer: threading.Timer | None = None

    def reset_for_new_session(self):
        """Reset state for a new session."""
        self.full_response_text = ""
        self.allow_mic_input = True
        self.assistant_speaking = False
        self.response_active = False
        self._turn_announced = False
        self._saw_transcript_delta = False
        self._turn_had_speech = False
        self._active_transcript_stream = None
        self._added_done_text = False
        self._saw_follow_up_call = False
        self._triggered_new_response = False
        self.follow_up_retry_count = 0
        self._skip_post_response_once = False
        self._last_heuristic_signature = None
        self._last_user_turn_meaningful = False
        self.follow_up_expected = False
        self.follow_up_prompt = None
        self._ignore_next_short_audio_response = False
        self._pending_input_audio_chunks = 0
        self._last_committed_audio_chunks = 0
        self._pending_loud_audio_chunks = 0
        self._last_committed_loud_audio_chunks = 0
        self._pending_peak_rms = 0.0
        self._last_committed_peak_rms = 0.0
        self._current_input_had_server_speech = False
        self._last_committed_had_server_speech = False
        self._server_input_speaking = False
        self._cancel_head_retract_timer()

    def on_response_created(self):
        """Handle response.created event."""
        self.response_active = True
        self.assistant_speaking = True  # Block mic immediately when response starts
        self.full_response_text = ""
        self._turn_announced = False
        self._saw_transcript_delta = False
        self._turn_had_speech = False
        self.follow_up_expected = False
        self.follow_up_prompt = None
        self._active_transcript_stream = None
        self._added_done_text = False
        self._last_heuristic_signature = None
        # Don't reset _saw_follow_up_call here - it will be reset after decision is made
        self._triggered_new_response = False

    def on_input_speech_started(self):
        """Handle input_audio_buffer.speech_started event."""
        self._current_input_had_server_speech = True
        self._server_input_speaking = True
        # Server VAD detected actual speech; keep session alive for quiet users.
        self.update_activity()

    def on_input_speech_stopped(self):
        """Handle input_audio_buffer.speech_stopped event."""
        self._server_input_speaking = False
        self.update_activity()

    def on_audio_committed(self, chunks: int):
        """Handle input_audio_buffer.committed event."""
        self._last_committed_audio_chunks = self._pending_input_audio_chunks
        self._last_committed_loud_audio_chunks = self._pending_loud_audio_chunks
        self._last_committed_peak_rms = self._pending_peak_rms
        self._last_committed_had_server_speech = self._current_input_had_server_speech
        self._pending_input_audio_chunks = 0
        self._pending_loud_audio_chunks = 0
        self._pending_peak_rms = 0.0
        self._current_input_had_server_speech = False
        # A committed user turn should count as recent activity while model processing starts.
        self.update_activity()
        logger.verbose(
            f"Committed audio turn with {self._last_committed_audio_chunks} chunks "
            f"({self._last_committed_loud_audio_chunks} above threshold, "
            f"peak_rms={self._last_committed_peak_rms:.1f}, "
            f"server_speech={self._last_committed_had_server_speech}).",
            "🎚️",
        )

    def on_conversation_item_done(self, data: dict[str, Any]):
        """Handle conversation.item.done event."""
        item = data.get("item") or {}
        if item.get("role") != "user":
            return

        content = item.get("content") or []
        if not content:
            return

        has_meaningful_user_content = False
        for part in content:
            text_bits = [
                (part.get("text") or "").strip(),
                (part.get("input_text") or "").strip(),
                (part.get("transcript") or "").strip(),
            ]
            if any(text_bits):
                has_meaningful_user_content = True
                break

        # Ignore audio-only turns with no transcript (silence/noise),
        # and very short audio blips.
        # Note: transcript can be None when server-side transcription is unavailable,
        # so transcript absence alone is not enough to classify as noise.
        audio_turn_meaningful = False
        if all(part.get("type") == "input_audio" for part in content):
            has_transcript = any(
                (part.get("transcript") or "").strip() for part in content
            )
            total_chunks = self._last_committed_audio_chunks
            loud_chunks = self._last_committed_loud_audio_chunks
            peak_rms = float(self._last_committed_peak_rms or 0.0)
            loud_ratio = (loud_chunks / total_chunks) if total_chunks else 0.0
            soft_speech_floor = max(120.0, SILENCE_THRESHOLD * 0.15)
            has_soft_local_speech = peak_rms >= soft_speech_floor

            # Heuristic noise gate:
            # - very short turns are usually accidental
            # - long turns with almost no energy above threshold are typically room noise
            min_chunks_for_real_turn = 6  # ~240ms with 40ms chunks
            low_signal_noise = (
                total_chunks >= 20 and loud_chunks <= 2 and loud_ratio < 0.12
            )
            static_only_turn = (
                not has_transcript
                and total_chunks >= min_chunks_for_real_turn
                and loud_chunks == 0
            )
            very_low_conf_server_speech = (
                self._last_committed_had_server_speech
                and not has_transcript
                and total_chunks >= min_chunks_for_real_turn
                and loud_chunks <= 1
                and loud_ratio < 0.05
            )
            should_ignore = total_chunks < min_chunks_for_real_turn or low_signal_noise
            if static_only_turn or very_low_conf_server_speech:
                should_ignore = True
            # If local RMS indicates soft but real speech, don't classify it as static.
            if (
                should_ignore
                and has_soft_local_speech
                and total_chunks >= min_chunks_for_real_turn
            ):
                should_ignore = False
            # Follow-up turns can be clipped at onset; if server VAD positively
            # detected speech and we captured at least a few chunks, treat it as
            # meaningful even when the short-turn heuristic would ignore it.
            if (
                should_ignore
                and self._last_committed_had_server_speech
                and total_chunks >= 3
                and loud_chunks >= 2
                and not static_only_turn
                and not very_low_conf_server_speech
            ):
                should_ignore = False
            if should_ignore:
                self._ignore_next_short_audio_response = True
                logger.info(
                    f"Ignoring non-speech audio turn "
                    f"({total_chunks} chunks, "
                    f"{loud_chunks} above threshold, "
                    f"ratio={loud_ratio:.2f}, "
                    f"peak_rms={peak_rms:.1f}, "
                    f"soft_speech_floor={soft_speech_floor:.1f}, "
                    f"server_speech={self._last_committed_had_server_speech}, "
                    f"has_transcript={has_transcript}, "
                    f"low_signal_noise={low_signal_noise}, "
                    f"static_only_turn={static_only_turn}, "
                    f"very_low_conf_server_speech={very_low_conf_server_speech}, "
                    f"has_soft_local_speech={has_soft_local_speech}).",
                    "🔇",
                )
            else:
                # Audio-only turns can still be meaningful even without transcript
                # (e.g. provider transcription missing, but local/server VAD shows speech).
                audio_turn_meaningful = bool(
                    has_transcript
                    or has_soft_local_speech
                    or loud_chunks >= 2
                    or (self._last_committed_had_server_speech and total_chunks >= 8)
                )

        # Only confirmed text/transcript input resets retry budget.
        # Keep prior `True` if another event already confirmed the turn.
        confirmed_meaningful_input = (
            self._last_user_turn_meaningful
            or has_meaningful_user_content
            or audio_turn_meaningful
        )
        if confirmed_meaningful_input:
            self.follow_up_retry_count = 0
        self._last_user_turn_meaningful = confirmed_meaningful_input

    def on_transcript_delta(self, stream_type: str, delta: str):
        """Handle transcript delta events."""
        # Choose a single transcript stream per turn to avoid duplicates
        if stream_type.startswith(
            "response.output_audio_transcript"
        ) or stream_type.startswith("response.audio_transcript"):
            stream = "audio"
        else:
            stream = "text"

        if self._active_transcript_stream is None:
            self._active_transcript_stream = stream
        elif stream != self._active_transcript_stream:
            return

        self._turn_had_speech = True
        self._saw_transcript_delta = True
        self.assistant_speaking = True
        self.allow_mic_input = False

        if not self._turn_announced:
            self.set_speaking_state()
            logger.info("Billy: ", "🐟")
            self._turn_announced = True

        self.full_response_text += delta

    def on_transcript_done(self, data: dict[str, Any]):
        """Handle transcript done events."""
        transcript = data.get("transcript") or data.get("text") or ""
        if transcript and not self._saw_transcript_delta and not self._added_done_text:
            self.full_response_text += transcript
            self._added_done_text = True
        self.full_response_text += "\n\n"
        logger.verbose(f"Transcript completed: {transcript!r}", "📝")

    def on_response_done(self):
        """Handle response.done event."""
        self.response_active = False
        self.assistant_speaking = False
        self.allow_mic_input = True

    def should_ignore_short_response(self) -> bool:
        """Check if we should ignore the next short audio response."""
        if self._ignore_next_short_audio_response:
            self._ignore_next_short_audio_response = False
            return True
        return False

    def wants_follow_up_heuristic(self) -> bool:
        """Check if response text suggests a follow-up is expected.

        If Billy asks a question, we expect the USER to follow up (respond),
        so we return True to keep the mic open.
        """
        txt = (self.full_response_text or "").strip()
        has_question = any(ch in txt for ch in ("?", "¿", "？", "؟", "‽"))
        signature = (txt, has_question)
        if signature != self._last_heuristic_signature:
            logger.verbose(
                f"Heuristic check: text='{txt}' | has_question={has_question}",
                "🔍",
            )
            self._last_heuristic_signature = signature
        # If Billy asks a question, keep mic open for user to respond
        return has_question

    def is_assistant_turn(self) -> bool:
        """Check if it's currently the assistant's turn."""
        return self.session.session_active.is_set() and self.assistant_speaking

    def is_user_turn(self) -> bool:
        """Check if it's currently the user's turn."""
        return (
            self.session.session_active.is_set()
            and self.allow_mic_input
            and not self.assistant_speaking
        )

    def set_listening_state(self):
        """Set Billy to listening state."""
        self._cancel_head_retract_timer()
        move_head("on")
        set_led_mode("listening")
        mqtt_publish("billy/state", "listening")

    def set_speaking_state(self):
        """Set Billy to speaking state."""
        self._schedule_head_retract()
        set_led_mode("idle")
        mqtt_publish("billy/state", "speaking")

    def set_idle_state(self):
        """Set Billy to idle state."""
        self._cancel_head_retract_timer()
        move_head("off")
        set_led_mode("idle")
        mqtt_publish("billy/state", "idle")

    def _cancel_head_retract_timer(self):
        """Cancel any pending delayed head retract."""
        if self._head_retract_timer:
            self._head_retract_timer.cancel()
            self._head_retract_timer = None

    def _schedule_head_retract(self):
        """Retract head asynchronously so speaking start is never blocked."""
        self._cancel_head_retract_timer()

        # 0 or negative means retract immediately (legacy behavior).
        if HEAD_RETRACT_DELAY_SECONDS <= 0:
            move_head("off")
            return

        self._head_retract_timer = threading.Timer(
            HEAD_RETRACT_DELAY_SECONDS, lambda: move_head("off")
        )
        self._head_retract_timer.daemon = True
        self._head_retract_timer.start()

    def increment_mic_chunks(self):
        """Increment pending input audio chunks counter."""
        self._pending_input_audio_chunks += 1

    def increment_loud_mic_chunks(self):
        """Increment pending audio chunks that are above local silence threshold."""
        self._pending_loud_audio_chunks += 1

    def observe_rms(self, rms: float):
        """Track peak RMS for the current pending input turn."""
        if rms > self._pending_peak_rms:
            self._pending_peak_rms = float(rms)

    def update_activity(self):
        """Update last activity timestamp."""
        self.session.last_activity[0] = time.time()

    def increment_follow_up_retry(self):
        """Increment follow-up retry counter."""
        self.follow_up_retry_count += 1

    def mark_user_turn_meaningful(self):
        """Mark the last user turn as meaningful and reset retry budget."""
        self._last_user_turn_meaningful = True
        self.follow_up_retry_count = 0
