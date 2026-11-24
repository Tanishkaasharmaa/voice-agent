"""Floating widget — a compact 40x20 pill with rounded ends that tracks the
agent state and reveals a settings gear on hover.

Active states animate a small glyph inside the pill:
  * LISTENING          — three dots that breathe in sequence
  * THINKING           — an ellipsis that cycles  ".  ..  ...  .."
  * SPEAKING           — a ") )) ))) ))" swell-and-shrink wave
  * TRANSCRIBING / EXECUTING_TOOL — a pulsing ellipsis
  * ERROR              — a "!" marker

The pill and the gear share one widget rect, so moving from the pill toward
the gear never leaves the hover region (the gear cannot vanish mid-approach).
The tooltip is intentionally omitted: a popup appearing under the cursor
steals the leave event and makes the gear flicker away.

Animations are a single QTimer-driven paint (low CPU).
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QCursor, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from ui.styles import STATE_COLORS

PILL_W = 40
PILL_H = 20
PILL_RADIUS = 10
GEAR_SIZE = 16
CLOSE_SIZE = 16

WIDGET_W = 82
WIDGET_H = 24

# The pill sits centered while idle, then slides left to make room for the
# gear and close buttons that fade in on hover.
PILL_X_IDLE = (WIDGET_W - PILL_W) / 2  # 21
PILL_X_HOVER = 2
GEAR_X = 44
CLOSE_X = GEAR_X + GEAR_SIZE + 2       # 62


class FloatingWidget(QWidget):
    settings_clicked = Signal()
    close_clicked = Signal()            # quit the whole program
    auto_listen_requested = Signal()    # double-click: continuous auto-stop listening
    position_changed = Signal(object)   # QPoint — while dragging / on release

    def __init__(self, settings=None) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool)
        self.settings = settings
        self._state = "IDLE"
        self._t = 0.0             # animation clock for glyphs
        self._hovered = False
        self._reveal = 0.0        # 0..1 gear reveal (animated)
        self._drag_offset: QPoint | None = None

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(WIDGET_W, WIDGET_H)
        self._apply_position()

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ------------------------------------------------------------- position

    def _apply_position(self) -> None:
        if self.settings is not None:
            x, y = self.settings.get("general.widget_position", [20, 120])
            self.move(int(x), int(y))

    def save_position(self) -> None:
        if self.settings is not None:
            pos = self.pos()
            self.settings.set("general.widget_position", [pos.x(), pos.y()])
            self.settings.save()

    # ----------------------------------------------------------------- state

    def set_state(self, state: str) -> None:
        self._state = state
        self.update()

    # --------------------------------------------------------- hover / drag

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()

    def leaveEvent(self, event) -> None:
        # Don't trust leave events blindly (overlapping translucent windows
        # and quick cursor crossings can fire them spuriously). Re-check the
        # real cursor position after a short grace period before hiding the
        # buttons — otherwise they can vanish mid-approach.
        QTimer.singleShot(250, self._refresh_hover)

    def _refresh_hover(self) -> None:
        under = self.rect().contains(self.mapFromGlobal(QCursor.pos()))
        if under != self._hovered:
            self._hovered = under
            self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._hovered:
                if self._close_rect().contains(event.position()):
                    self.close_clicked.emit()
                    event.accept()
                    return
                if self._gear_rect().contains(event.position()):
                    self.settings_clicked.emit()
                    event.accept()
                    return
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            self.position_changed.emit(self.pos())
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_offset is not None:
            self._drag_offset = None
            self.save_position()
            self.position_changed.emit(self.pos())
            event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        # Double-click no longer hides the island — it starts continuous
        # auto-stop listening (the app wires this signal).
        self.auto_listen_requested.emit()

    # ------------------------------------------------------------ animation

    def _tick(self) -> None:
        self._t += 0.15
        target = 1.0 if self._hovered else 0.0
        self._reveal += (target - self._reveal) * 0.25
        self.update()

    # ---------------------------------------------------------------- rects

    def _pill_rect(self) -> QRectF:
        x = PILL_X_IDLE + (PILL_X_HOVER - PILL_X_IDLE) * self._reveal
        y = (WIDGET_H - PILL_H) / 2
        return QRectF(x, y, PILL_W, PILL_H)

    def _gear_rect(self) -> QRectF:
        y = (WIDGET_H - GEAR_SIZE) / 2
        return QRectF(GEAR_X, y, GEAR_SIZE, GEAR_SIZE)

    def _close_rect(self) -> QRectF:
        y = (WIDGET_H - CLOSE_SIZE) / 2
        return QRectF(CLOSE_X, y, CLOSE_SIZE, CLOSE_SIZE)

    # ----------------------------------------------------------------- paint

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            color_hex, _label = STATE_COLORS.get(self._state, STATE_COLORS["IDLE"])
            base = QColor(color_hex)

            pill = self._pill_rect()
            path = QPainterPath()
            path.addRoundedRect(pill, PILL_RADIUS, PILL_RADIUS)
            painter.fillPath(path, base)

            # Faint rim so the pill reads on any background.
            painter.setPen(QPen(QColor(255, 255, 255, 28), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

            self._draw_glyph(painter, pill)

            # Gear + close — kept inside the same widget rect so they never
            # vanish while the cursor moves toward them.
            if self._reveal > 0.02:
                slide = (1.0 - self._reveal) * 10.0
                painter.save()
                painter.setOpacity(self._reveal)
                gear = self._gear_rect()
                gear.moveLeft(gear.left() + slide)
                self._draw_gear(painter, gear)
                close_btn = self._close_rect()
                close_btn.moveLeft(close_btn.left() + slide)
                self._draw_close(painter, close_btn)
                painter.restore()
        finally:
            painter.end()

    # ---------------------------------------------------------------- glyphs

    def _glyph_font(self) -> QFont:
        font = QFont()
        font.setPixelSize(10)
        font.setBold(True)
        return font

    def _draw_glyph(self, painter: QPainter, pill: QRectF) -> None:
        if self._state == "LISTENING":
            self._draw_dots(painter, pill)
        elif self._state == "THINKING":
            steps = [".", "..", "...", ".."]
            text = steps[int(self._t / 3) % len(steps)]
            self._draw_text(painter, pill, text)
        elif self._state == "SPEAKING":
            self._draw_wave(painter, pill)
        elif self._state in ("TRANSCRIBING", "EXECUTING_TOOL"):
            self._draw_text(painter, pill, "\u2026", pulse=True)
        elif self._state == "ERROR":
            self._draw_text(painter, pill, "!", pulse=True)

    def _draw_text(self, painter: QPainter, pill: QRectF, text: str,
                   pulse: bool = False) -> None:
        painter.setFont(self._glyph_font())
        color = QColor("#ffffff")
        if pulse:
            color.setAlphaF(0.55 + 0.45 * (0.5 + 0.5 * math.sin(self._t * 6)))
        painter.setPen(QPen(color))
        painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_dots(self, painter: QPainter, pill: QRectF) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        r = 1.6
        gap = 6.0
        total = 2 * gap
        cx = pill.center().x()
        cy = pill.center().y()
        for i in range(3):
            alpha = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(self._t * 5 - i * 2.1))
            color = QColor("#ffffff")
            color.setAlphaF(alpha)
            painter.setBrush(color)
            painter.drawEllipse(QPointF(cx - total / 2 + i * gap, cy), r, r)

    def _draw_wave(self, painter: QPainter, pill: QRectF) -> None:
        # A speaker-style swell that grows and shrinks smoothly:
        # )  ))  )))  ))  )  ... (no jump back to a single paren)
        steps = [")", "))", ")))", "))"]
        text = steps[int(self._t / 4) % len(steps)]
        self._draw_text(painter, pill, text)

    # ------------------------------------------------------------------ gear

    def _draw_gear(self, painter: QPainter, rect: QRectF) -> None:
        center = rect.center()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#e2e8f0"))
        # Teeth around the rim.
        painter.save()
        painter.translate(center)
        for _ in range(8):
            painter.rotate(45)
            painter.drawRoundedRect(QRectF(-1.5, -7.0, 3.0, 4.0), 1.0, 1.0)
        painter.restore()
        # Rim and hole.
        painter.drawEllipse(center, 5.0, 5.0)
        painter.setBrush(QColor("#0f172a"))
        painter.drawEllipse(center, 2.2, 2.2)

    def _draw_close(self, painter: QPainter, rect: QRectF) -> None:
        center = rect.center()
        pen = QPen(QColor("#e2e8f0"), 1.8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        r = 4.6
        painter.drawLine(QPointF(center.x() - r, center.y() - r),
                         QPointF(center.x() + r, center.y() + r))
        painter.drawLine(QPointF(center.x() - r, center.y() + r),
                         QPointF(center.x() + r, center.y() - r))