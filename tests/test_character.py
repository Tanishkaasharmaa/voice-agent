"""Tests for the character animator and mouth-level smoothing (no Qt)."""

from ui.character_widget import CharacterAnimator, smooth_level


def make_animator(state="IDLE"):
    a = CharacterAnimator()
    a.set_state(state)
    return a


def test_smooth_level_rises_fast_falls_slow():
    # attack: approaches target quickly
    v = 0.0
    for _ in range(20):
        v = smooth_level(v, 1.0)
    assert v > 0.99
    # release: decays slowly
    v = 1.0
    for _ in range(5):
        v = smooth_level(v, 0.0)
    assert 0.3 < v < 0.9


def test_mouth_only_moves_while_speaking():
    a = make_animator("IDLE")
    a.set_mouth_level(1.0)
    a.tick()
    assert a.mouth_open() == 0.0
    a.set_state("SPEAKING")
    a.set_mouth_level(1.0)
    a.tick()
    assert a.mouth_open() > 0.0


def test_mouth_clamped_to_unit_range():
    a = make_animator("SPEAKING")
    a.set_mouth_level(100.0)   # absurd RMS must clamp
    for _ in range(60):
        a.tick()
        assert 0.0 <= a.mouth_open() <= 1.0


def test_mouth_decays_when_events_stop():
    a = make_animator("SPEAKING")
    a.set_mouth_level(1.0)
    for _ in range(10):
        a.tick()
    opened = a.mouth_open()
    for _ in range(200):       # no further events; target decays to ~0
        a.tick()
    assert a.mouth_open() < opened * 0.05


def test_set_same_state_keeps_mouth_target():
    a = make_animator("SPEAKING")
    a.set_mouth_level(1.0)
    a.set_state("SPEAKING")    # no-op transition
    a.tick()
    assert a.mouth_open() > 0.0


def test_eye_open_blinks_between_zero_and_one():
    a = make_animator("IDLE")
    values = [a.eye_open()]
    for _ in range(600):
        a.tick(0.033)
        values.append(a.eye_open())
        assert 0.0 < a.eye_open() <= 1.0
    # a blink must have occurred within ~20 s of ticks
    assert min(values) < 0.4


def test_look_offsets_per_state():
    idle = make_animator("IDLE").look()
    thinking = make_animator("THINKING").look()
    assert idle != thinking
    for x, y in (idle, thinking):
        assert -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0


def test_animation_clock_advances():
    a = CharacterAnimator()
    before = a.t
    a.tick(0.033)
    assert a.t > before


def test_hands_ease_toward_state_pose():
    a = CharacterAnimator()
    start = a.hands
    a.set_state("LISTENING")
    for _ in range(80):          # ~2.6 s of ticks
        a.tick()
    target_l, target_r = __import__("ui.character_widget",
                                    fromlist=["HAND_POSES"]).HAND_POSES["LISTENING"]
    # close to target after easing
    assert abs(a.hands[1][0] - target_r[0]) < 0.02
    assert abs(a.hands[1][1] - target_r[1]) < 0.02
    assert a.hands != start
