"""Hotkey manager tests.

The manager must never return ``False`` from its callbacks (pynput stops
the listener on that) — suppression happens through ``_handle_key`` and the
``win32_event_filter`` path instead.
"""

from hotkeys.manager import (
    HotkeyManager,
    _VK_LCONTROL,
    _VK_LWIN,
    _VK_SPACE,
    _WM_KEYDOWN,
    _WM_KEYUP,
)


class Key:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Key.{self.name}"


class FakeEvent:
    """Minimal stand-in for pynput's KBDLLHOOKSTRUCT."""

    def __init__(self, vk_code):
        self.vkCode = vk_code


class FakeListener:
    def __init__(self, on_press=None, on_release=None, win32_event_filter=None):
        self.on_press = on_press
        self.on_release = on_release
        self.win32_event_filter = win32_event_filter
        self.running = True
        self.suppressed = 0

    def start(self):
        pass

    def stop(self):
        self.running = False

    def suppress_event(self):
        self.suppressed += 1


def make_manager():
    manager = HotkeyManager(listener_factory=lambda **kw: FakeListener(**kw))
    events = []
    manager.on_ptt_press = lambda: events.append("press")
    manager.on_ptt_release = lambda: events.append("release")
    manager.on_auto_stop = lambda: events.append("auto")
    manager.events = events
    manager.start()
    return manager


# ------------------------------------------------------------- callbacks


def test_callbacks_never_return_false():
    # pynput stops the listener if a callback returns False — the whole
    # point of the rewrite is that callbacks always return None.
    manager = make_manager()
    assert manager._on_press(Key("ctrl_l")) is None
    assert manager._on_release(Key("ctrl_l")) is None
    assert manager._on_press(Key("cmd")) is None
    assert manager._on_press(Key("space")) is None


def test_push_to_talk_press_and_release_via_callbacks():
    manager = make_manager()
    manager._on_press(Key("ctrl_l"))
    assert manager.events == []
    manager._on_press(Key("cmd"))
    assert manager.events == ["press"]
    manager._on_release(Key("ctrl_l"))
    assert manager.events == ["press", "release"]


def test_auto_stop_with_space_via_callbacks():
    manager = make_manager()
    manager._on_press(Key("ctrl_r"))
    manager._on_press(Key("cmd"))
    assert manager.events == ["press"]
    manager._on_press(Key("space"))
    assert manager.events == ["press", "auto"]


def test_plain_win_key_not_swallowed():
    manager = make_manager()
    # Win pressed alone must propagate (no suppression, no listening).
    assert manager._handle_key("win", True) is False
    assert manager.events == []


def test_reset_keys_clears_state():
    manager = make_manager()
    manager._on_press(Key("ctrl_l"))
    manager._on_press(Key("cmd"))
    assert manager.events == ["press"]
    manager.reset_keys()
    manager._on_release(Key("ctrl_l"))
    assert manager.events == ["press"]  # no spurious release event


# ------------------------------------------------------- win32 filter


def test_event_filter_engages_ptt_and_suppresses_win():
    manager = make_manager()
    listener = manager._listener
    # Ctrl down: not suppressed, no event yet.
    assert manager._event_filter(_WM_KEYDOWN, FakeEvent(_VK_LCONTROL)) is True
    assert listener.suppressed == 0
    assert manager.events == []
    # Win down inside the combo: pressed and suppressed system wide.
    assert manager._event_filter(_WM_KEYDOWN, FakeEvent(_VK_LWIN)) is True
    assert manager.events == ["press"]
    assert listener.suppressed == 1
    # Ctrl up: released, not suppressed.
    assert manager._event_filter(_WM_KEYUP, FakeEvent(_VK_LCONTROL)) is True
    assert manager.events == ["press", "release"]


def test_event_filter_auto_stop_suppresses_space():
    manager = make_manager()
    listener = manager._listener
    manager._event_filter(_WM_KEYDOWN, FakeEvent(_VK_LCONTROL))
    manager._event_filter(_WM_KEYDOWN, FakeEvent(_VK_LWIN))
    assert manager.events == ["press"]
    assert manager._event_filter(_WM_KEYDOWN, FakeEvent(_VK_SPACE)) is True
    assert manager.events == ["press", "auto"]
    assert listener.suppressed == 2  # win + space


def test_event_filter_plain_win_not_suppressed():
    manager = make_manager()
    listener = manager._listener
    assert manager._event_filter(_WM_KEYDOWN, FakeEvent(_VK_LWIN)) is True
    assert listener.suppressed == 0
    assert manager.events == []


def test_win_before_ctrl_order_independence():
    manager = make_manager()
    # Win pressed first, then Ctrl — combo still engages.
    manager._on_press(Key("cmd"))
    assert manager.events == []
    manager._on_press(Key("ctrl_l"))
    assert manager.events == ["press"]


def test_start_and_stop_lifecycle():
    manager = make_manager()
    assert manager.running is True
    manager.stop()
    assert manager.running is False


def test_start_returns_false_without_pynput(monkeypatch):
    manager = HotkeyManager(listener_factory=None)

    def boom(*a, **k):
        raise ImportError("pynput missing")

    monkeypatch.setattr(
        "builtins.__import__",
        lambda name, *a, **k: boom() if name == "pynput" else __import__(name, *a, **k),
    )
    assert manager.start() is False
    assert manager.running is False
