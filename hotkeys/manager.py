"""Global hotkey manager (pynput).

Modes:
* MODE 1 — push-to-talk:  CTRL + WINDOWS  (press = listen, release = stop)
* MODE 2 — auto-stop:     CTRL + WINDOWS + SPACE (VAD ends the utterance)

Start-menu suppression is implemented with pynput's ``win32_event_filter``
hook: while the push-to-talk combo is engaged the Win keydown is suppressed
system wide (via ``suppress_event()``), so the Start menu never opens even if
the user releases CTRL before WINDOWS. The paired Win keyup still reaches the
hook (we pass it through, releasing the combo), so the key state never
sticks. Plain Win-key use is untouched.

IMPORTANT: callbacks must never return ``False`` — in pynput, a callback
returning ``False`` stops the listener entirely (it raises ``StopException``
internally). All logic lives in the event filter, which is *not* subject to
that wrapper; the ``on_press``/``on_release`` callbacks are only a fallback
for backends without the win32 filter.
"""

from __future__ import annotations

import threading
from typing import Callable

MOD_PUSH_TO_TALK = ("ctrl", "win")
MOD_AUTO_STOP = ("ctrl", "win", "space")

# Windows messages and virtual-key codes (WH_KEYBOARD_LL).
_WM_KEYDOWN = 0x0100
_WM_SYSKEYDOWN = 0x0104
_WM_KEYUP = 0x0101
_WM_SYSKEYUP = 0x0105
_VK_CONTROL = 0x11
_VK_LCONTROL = 0xA2
_VK_RCONTROL = 0xA3
_VK_LWIN = 0x5B
_VK_RWIN = 0x5C
_VK_SPACE = 0x20

_VK_NAMES = {
    _VK_CONTROL: "ctrl",
    _VK_LCONTROL: "ctrl",
    _VK_RCONTROL: "ctrl",
    _VK_LWIN: "win",
    _VK_RWIN: "win",
    _VK_SPACE: "space",
}


def _normalize(key) -> str | None:
    """Normalize a pynput ``Key``/``KeyCode`` to a combo name (or None)."""
    try:
        if hasattr(key, "name"):
            name = key.name.lower()
            if name in ("ctrl_l", "ctrl_r", "ctrl"):
                return "ctrl"
            if name in ("cmd", "cmd_l", "cmd_r", "win", "win_l", "win_r"):
                return "win"
            if name == "space":
                return "space"
        return None
    except Exception:
        return None


class HotkeyManager:
    def __init__(self, push_to_talk: tuple = MOD_PUSH_TO_TALK,
                 auto_stop: tuple = MOD_AUTO_STOP,
                 listener_factory=None) -> None:
        self.push_to_talk = set(push_to_talk)
        self.auto_stop = set(auto_stop)
        self._listener_factory = listener_factory  # resolved lazily in start()
        self._lock = threading.Lock()
        self._listener = None

        # Per-session key state (mutated only under _lock).
        self._pressed: set[str] = set()      # tracked modifiers actually seen
        self._win_pending = False            # Win down, not yet in a combo
        self._win_engaged = False            # Win part of an active combo
        self._space_down = False             # Space down during the combo
        self._auto_fired = False             # auto-stop fired this session

        self.on_ptt_press: Callable[[], None] | None = None
        self.on_ptt_release: Callable[[], None] | None = None
        self.on_auto_stop: Callable[[], None] | None = None

    # -------------------------------------------------------------- startup

    def start(self) -> bool:
        try:
            factory = self._listener_factory
            if factory is None:
                from pynput import keyboard  # lazy — pynput may be absent

                factory = keyboard.Listener
            self._listener = factory(
                on_press=self._on_press,
                on_release=self._on_release,
                win32_event_filter=self._event_filter,
            )
            self._listener.start()
            return True
        except Exception:
            return False

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

    @property
    def running(self) -> bool:
        return bool(self._listener and self._listener.running)

    # ------------------------------------------------------------ callbacks
    # NOTE: these must NEVER return False — pynput treats a False return as
    # "stop the listener". The win32_event_filter does the real work; the
    # callbacks re-feed the same handler as a fallback for backends without
    # the filter, and _handle_key is idempotent so double-processing is a
    # no-op.

    def _on_press(self, key) -> None:
        name = _normalize(key)
        if name:
            self._handle_key(name, is_down=True)

    def _on_release(self, key) -> None:
        name = _normalize(key)
        if name:
            self._handle_key(name, is_down=False)

    # ------------------------------------------------------ win32 filter

    def _event_filter(self, msg, data):
        """pynput ``win32_event_filter`` — runs on the hook thread for every
        key event, before dispatch. Returns True to pass the event on; calls
        ``suppress_event()`` to swallow it system wide.
        """
        name = _VK_NAMES.get(data.vkCode)
        if name is None:
            return True
        is_down = msg in (_WM_KEYDOWN, _WM_SYSKEYDOWN)
        suppress = self._handle_key(name, is_down)
        if suppress and self._listener is not None:
            # Raises SystemHook.SuppressException, which the hook procedure
            # converts into a system-wide suppression of this event.
            self._listener.suppress_event()
        return True

    # --------------------------------------------------------- core logic

    def _ptt_engaged(self) -> bool:
        """True while the push-to-talk combo (CTRL+WIN) is physically active."""
        return "ctrl" in self._pressed and self._win_engaged

    def _handle_key(self, name: str, is_down: bool) -> bool:
        """Update combo state for one key event.

        Returns True when the OS event should be suppressed system wide
        (the Win key during an engaged combo, or Space in auto-stop).
        Idempotent: safe to call twice for the same event (filter + callback).
        """
        with self._lock:
            if name == "ctrl":
                was = self._ptt_engaged()
                if is_down:
                    self._pressed.add("ctrl")
                    if self._win_pending:
                        # Win was pressed before Ctrl — engage from the
                        # pending Win press so key order does not matter.
                        self._win_pending = False
                        self._win_engaged = True
                else:
                    self._pressed.discard("ctrl")
                now = self._ptt_engaged()
                self._fire_transition(was, now)
                if not now:
                    # Combo session over — clear per-session state so no
                    # stale flags leak into the next press.
                    self._win_engaged = False
                    self._space_down = False
                    self._auto_fired = False
                return False

            if name == "win":
                if is_down:
                    if "ctrl" in self._pressed:
                        was = self._ptt_engaged()
                        self._win_engaged = True
                        self._win_pending = False
                        now = self._ptt_engaged()
                        self._fire_transition(was, now)
                        # Swallow the Win keydown: no Start menu even if CTRL
                        # is released first. The paired keyup still reaches
                        # the hook (see below) and is passed through, so the
                        # key state never sticks.
                        return True
                    self._win_pending = True  # plain Win press (Start opens)
                else:
                    self._win_pending = False
                    if self._win_engaged:
                        # Win released first (the keyup reached us even though
                        # the keydown was suppressed) — combo ends here.
                        self._win_engaged = False
                        was, now = True, False
                        self._fire_transition(was, now)
                        self._space_down = False
                        self._auto_fired = False
                return False

            if name == "space":
                if is_down:
                    was = self._ptt_engaged()
                    self._space_down = True
                    if was and not self._auto_fired and self.on_auto_stop:
                        self._auto_fired = True
                        self.on_auto_stop()
                        # Swallow the space so Win+Space never switches the
                        # keyboard layout while the combo is held.
                        return True
                else:
                    self._space_down = False
                return False

            return False

    def _fire_transition(self, was: bool, now: bool) -> None:
        if now and not was:
            if self.on_ptt_press:
                self.on_ptt_press()
        elif was and not now:
            if self.on_ptt_release:
                self.on_ptt_release()

    def reset_keys(self) -> None:
        """Clear pressed state (used on errors so a stuck modifier doesn't
        leave the app listening forever)."""
        with self._lock:
            self._pressed.clear()
            self._win_pending = False
            self._win_engaged = False
            self._space_down = False
            self._auto_fired = False
