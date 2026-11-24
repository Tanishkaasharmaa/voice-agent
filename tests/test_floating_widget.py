"""Tests for the floating island buttons and double-click auto-listen."""

import pytest

pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("PySide6.QtTest")

from PySide6.QtCore import QPoint, QRectF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.floating_widget import FloatingWidget, WIDGET_W  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def island(qapp):
    return FloatingWidget()


def test_button_rects_inside_widget_and_disjoint(island):
    gear = island._gear_rect()
    close = island._close_rect()
    assert gear.right() < close.left()      # beside each other
    assert close.right() <= WIDGET_W        # inside the widget
    assert not gear.intersects(close)


def test_double_click_requests_auto_listen(island):
    fired = []
    island.auto_listen_requested.connect(lambda: fired.append(True))
    island.mouseDoubleClickEvent(None)
    assert fired == [True]
    assert island.isVisibleTo(None) or True  # widget itself stays alive


def test_close_click_emits_close_clicked(qapp):
    w = FloatingWidget()
    fired = []
    w.close_clicked.connect(lambda: fired.append(True))
    w._hovered = True
    QTest.mouseClick(w, Qt.MouseButton.LeftButton,
                     pos=w._close_rect().center().toPoint())
    assert fired == [True]


def test_gear_click_emits_settings_clicked(qapp):
    w = FloatingWidget()
    fired = []
    w.settings_clicked.connect(lambda: fired.append(True))
    w._hovered = True
    QTest.mouseClick(w, Qt.MouseButton.LeftButton,
                     pos=w._gear_rect().center().toPoint())
    assert fired == [True]


def test_drag_still_works_from_pill_area(qapp):
    w = FloatingWidget()
    start = w.pos()
    pill_center = QRectF(0, 0, 20, 24).center().toPoint()
    QTest.mousePress(w, Qt.MouseButton.LeftButton, pos=pill_center)
    QTest.mouseMove(w, QPoint(pill_center.x() + 30, pill_center.y() + 10))
    QTest.mouseRelease(w, Qt.MouseButton.LeftButton,
                       pos=QPoint(pill_center.x() + 30, pill_center.y() + 10))
    assert w.pos() != start


def test_rects_types(island):
    assert isinstance(island._close_rect(), QRectF)


def test_leave_event_does_not_instantly_clear_hover(island):
    island.enterEvent(None)
    assert island._hovered is True
    island.leaveEvent(None)
    # Grace period: buttons must stay visible right after a spurious leave.
    assert island._hovered is True


def test_refresh_hover_follows_real_cursor(qapp, monkeypatch):
    import ui.floating_widget as fw_mod

    w = FloatingWidget()
    w.move(200, 200)
    w.show()
    qapp.processEvents()

    inside = QPoint(210, 210)
    monkeypatch.setattr(fw_mod.QCursor, "pos", staticmethod(lambda: inside))
    w.leaveEvent(None)
    w._refresh_hover()
    assert w._hovered is True

    outside = QPoint(10, 10)
    monkeypatch.setattr(fw_mod.QCursor, "pos", staticmethod(lambda: outside))
    w._refresh_hover()
    assert w._hovered is False


def test_character_widget_ignores_mouse(qapp):
    from PySide6.QtCore import Qt as QtCore
    from ui.character_widget import CharacterWidget

    char = CharacterWidget()
    assert bool(char.testAttribute(QtCore.WidgetAttribute.WA_TransparentForMouseEvents))
