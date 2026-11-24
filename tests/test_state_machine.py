import pytest

from app.events import EventBus
from app.state_machine import (
    ERROR, EXECUTING_TOOL, IDLE, LISTENING, SPEAKING, THINKING,
    TRANSCRIBING, InvalidTransition, StateMachine,
)


@pytest.fixture
def machine():
    bus = EventBus()
    sm = StateMachine(bus)
    events = []
    bus.subscribe("state", lambda _topic, state: events.append(state))
    sm.events = events
    return sm


def test_initial_state_is_idle(machine):
    assert machine.state == IDLE


def test_valid_pipeline(machine):
    machine.transition(LISTENING)
    machine.transition(TRANSCRIBING)
    machine.transition(THINKING)
    machine.transition(EXECUTING_TOOL)
    machine.transition(THINKING)
    machine.transition(SPEAKING)
    machine.transition(IDLE)
    assert machine.state == IDLE


def test_invalid_transition_raises(machine):
    with pytest.raises(InvalidTransition):
        machine.transition(SPEAKING)  # IDLE -> SPEAKING is not allowed
    assert machine.state == IDLE


def test_any_state_can_go_to_error(machine):
    machine.transition(LISTENING)
    machine.transition(ERROR)
    assert machine.state == ERROR
    machine.transition(IDLE)  # ERROR -> IDLE recovery
    assert machine.state == IDLE


def test_speaking_can_be_interrupted_to_listening(machine):
    machine.transition(THINKING)
    machine.transition(SPEAKING)
    machine.transition(LISTENING)  # interruption
    assert machine.state == LISTENING


def test_transcribing_and_thinking_can_be_interrupted_to_listening(machine):
    machine.transition(LISTENING)
    machine.transition(TRANSCRIBING)
    machine.transition(LISTENING)  # interruption during transcription
    assert machine.state == LISTENING

    machine.transition(TRANSCRIBING)
    machine.transition(THINKING)
    machine.transition(LISTENING)  # interruption during thinking
    assert machine.state == LISTENING


def test_state_events_published(machine):
    machine.transition(LISTENING)
    assert machine.events[-1] == LISTENING


def test_can_check_membership(machine):
    machine.transition(LISTENING)
    assert machine.is_in(LISTENING)
    assert not machine.is_in(TRANSCRIBING)


def test_reset_returns_to_idle(machine):
    machine.transition(THINKING)
    machine.reset()
    assert machine.state == IDLE


def test_set_state_is_idempotent_for_repeated_speaking():
    """Regression: the pipeline re-requests SPEAKING for every streamed
    sentence. The redundant transition must be a no-op -- the old fallback
    (reset + retry) published IDLE mid-speech and could not get back to
    SPEAKING (IDLE->SPEAKING is not a legal move), leaving the UI idle
    while audio kept playing."""
    from app.application import VoiceAgentApplication

    class _Harness:
        def __init__(self):
            self.state = StateMachine(EventBus())

    h = _Harness()
    events = []
    h.state.add_listener(lambda _o, n: events.append(n))

    VoiceAgentApplication._set_state(h, THINKING)
    VoiceAgentApplication._set_state(h, SPEAKING)   # first sentence
    VoiceAgentApplication._set_state(h, SPEAKING)   # second sentence
    VoiceAgentApplication._set_state(h, SPEAKING)   # third sentence

    assert h.state.state == SPEAKING
    assert events == [THINKING, SPEAKING]  # no spurious IDLE flash


def _app():
    from app.application import VoiceAgentApplication
    return VoiceAgentApplication


def _make_app_harness():
    from app.events import EventBus
    from app.state_machine import StateMachine

    class _FakeAudio:
        def __init__(self):
            self.started_with = []

        @property
        def mode(self):
            return "auto_stop"

        def start_listening(self, mode):
            self.started_with.append(mode)

    class _Harness:
        def __init__(self, voice_enabled=True):
            self.bus = EventBus()
            self.state = StateMachine(self.bus)
            self._voice_enabled = voice_enabled
            self._shutting_down = False
            self.audio_manager = _FakeAudio()

        def _start_listening(self, mode):
            # Mirrors the real guard: only start when not already listening.
            if self.state.is_in(LISTENING):
                return
            self.state.transition(LISTENING)
            self.audio_manager.start_listening(mode)

        def _set_state(self, state):
            if self.state.state == state:
                return
            try:
                self.state.transition(state)
            except Exception:
                pass

    return _Harness


def test_finish_turn_continuous_resumes_listening():
    VoiceAgentApplication = _app()
    h = _make_app_harness()()
    h.state.transition(THINKING)
    h._set_state(SPEAKING)
    VoiceAgentApplication._finish_turn(h, was_continuous=True)
    assert h.state.state == LISTENING              # back to listening...
    assert h.audio_manager.started_with == ["auto_stop"]


def test_finish_turn_ptt_goes_idle():
    VoiceAgentApplication = _app()
    h = _make_app_harness()()
    h.state.transition(THINKING)
    h._set_state(SPEAKING)
    VoiceAgentApplication._finish_turn(h, was_continuous=False)
    assert h.state.state == IDLE                   # push-to-talk ends idle
    assert h.audio_manager.started_with == []


def test_finish_turn_no_resume_when_voice_disabled():
    VoiceAgentApplication = _app()
    h = _make_app_harness()(voice_enabled=False)
    h.state.transition(THINKING)
    h._set_state(SPEAKING)
    VoiceAgentApplication._finish_turn(h, was_continuous=True)
    assert h.state.state == IDLE
    assert h.audio_manager.started_with == []


def test_finish_turn_does_not_stomp_active_session():
    """User re-triggered listening while the turn was finishing — the
    pipeline must not yank it through IDLE."""
    VoiceAgentApplication = _app()
    h = _make_app_harness()()
    h.state.transition(THINKING)
    h._set_state(SPEAKING)
    h.state.transition(LISTENING)                  # new session mid-turn
    events = []
    h.bus.subscribe("state", lambda _t, s: events.append(s))
    VoiceAgentApplication._finish_turn(h, was_continuous=True)
    assert h.state.state == LISTENING
    assert events == []                            # no spurious transitions
