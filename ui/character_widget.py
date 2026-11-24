"""Animated procedural anime-style characters rendered above the island.

``CharacterAnimator`` is pure logic (no Qt): it holds the agent state and a
smoothed mouth level, and computes per-frame animation parameters (breathing,
blink, sway, gaze, eased hand poses). ``CharacterWidget`` paints those
parameters with QPainter on one 33 ms QTimer — same low-CPU recipe as
FloatingWidget.

Characters are **presets** (``CHARACTER_PRESETS``): palette + hairstyle +
outfit combos painted procedurally, so switching characters needs no model
downloads and stays fully offline. ``render_preview`` paints any preset into
a QImage for the Settings dialog thumbnails.

Window geometry mirrors ui/floating_widget.py: frameless, translucent,
always-on-top Qt.Tool window positioned relative to the island via
``character.offset`` from config.
"""

from __future__ import annotations

import math
import random

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPen, QImage
from PySide6.QtWidgets import QWidget

CHAR_W = 140
CHAR_H = 200


def smooth_level(current: float, target: float,
                 attack: float = 0.45, release: float = 0.12) -> float:
    """One EMA step; rises fast (attack) so the mouth snaps open on speech,
    falls slowly (release) so it closes naturally."""
    k = attack if target > current else release
    return current + (target - current) * k


# Hand positions per state, normalized to the widget rect (x, y).
HAND_POSES = {
    "IDLE": ((0.25, 0.66), (0.75, 0.66)),
    "LISTENING": ((0.25, 0.66), (0.72, 0.36)),      # right hand cupped to ear
    "TRANSCRIBING": ((0.26, 0.67), (0.70, 0.38)),
    "THINKING": ((0.27, 0.68), (0.58, 0.42)),       # finger to chin
    "EXECUTING_TOOL": ((0.34, 0.56), (0.66, 0.56)), # hands forward, busy
    "SPEAKING": ((0.20, 0.60), (0.80, 0.60)),       # expressive, open
    "ERROR": ((0.22, 0.48), (0.78, 0.48)),          # hands raised, worried
}

# ------------------------------------------------------------- character
#                                                    presets (the waifus)

SKIN = "#f5d9c8"
SKIN_SHADE = "#eac4b0"

CHARACTER_PRESETS: dict[str, dict] = {
    "luna": {
        "label": "Luna",
        "style": "twin_tails",
        "hair_dark": "#3a2c50", "hair_light": "#54406e",
        "eye": "#7c5cff", "dress_top": "#463a75", "dress_bot": "#332a57",
    },
    "sakura": {
        "label": "Sakura",
        "style": "long",
        "hair_dark": "#d16a8c", "hair_light": "#f2a3bd",
        "eye": "#e35d7c", "dress_top": "#8c4a5e", "dress_bot": "#6d3548",
    },
    "yuki": {
        "label": "Yuki",
        "style": "long",
        "hair_dark": "#9aa7bd", "hair_light": "#dde6f2",
        "eye": "#57b8ff", "dress_top": "#4a6fa5", "dress_bot": "#31508a",
    },
    "aoi": {
        "label": "Aoi",
        "style": "bob",
        "hair_dark": "#2b4a7a", "hair_light": "#4a7ab5",
        "eye": "#37d0c4", "dress_top": "#2f3b52", "dress_bot": "#20293a",
    },
    "mikan": {
        "label": "Mikan",
        "style": "ponytail",
        "hair_dark": "#b56a2a", "hair_light": "#f0a24f",
        "eye": "#e8a020", "dress_top": "#a05a2c", "dress_bot": "#7c401f",
    },
}

DEFAULT_CHARACTER = "luna"


class CharacterAnimator:
    """State + smoothed mouth level -> animation parameters. No Qt."""

    def __init__(self) -> None:
        self.state = "IDLE"
        self.t = 0.0                       # seconds since start
        self._mouth_target = 0.0
        self.mouth = 0.0                   # smoothed 0..1
        self._blink_in = random.uniform(2.0, 6.0)   # seconds to next blink
        self._blink_left = 0.0             # >0 while blinking
        self.hands = HAND_POSES["IDLE"]    # eased toward HAND_POSES[state]

    # ------------------------------------------------------------- inputs

    def set_state(self, state: str) -> None:
        if state == self.state:
            return
        self.state = state
        if state != "SPEAKING":
            self._mouth_target = 0.0

    def set_mouth_level(self, level: float) -> None:
        """Raw RMS from the TTS speaker (roughly 0..1)."""
        if self.state != "SPEAKING":
            return
        self._mouth_target = max(0.0, min(1.0, float(level) * 4.0))

    # -------------------------------------------------------------- tick

    def tick(self, dt: float = 0.033) -> None:
        self.t += dt
        # Decay toward closed even if no zero-level events arrive between
        # sentences, then apply smoothing.
        self._mouth_target *= 0.90
        self.mouth = smooth_level(self.mouth, self._mouth_target)

        if self._blink_left > 0.0:
            self._blink_left -= dt
        else:
            self._blink_in -= dt
            if self._blink_in <= 0.0:
                self._blink_left = 0.15
                self._blink_in = random.uniform(2.0, 6.0)

        # Ease hands toward this state's pose.
        target_l, target_r = HAND_POSES.get(self.state, HAND_POSES["IDLE"])
        hl, hr = self.hands
        ease = lambda cur, tgt: tuple(c + (n - c) * 0.12 for c, n in zip(cur, tgt))
        self.hands = (ease(hl, target_l), ease(hr, target_r))

    # ------------------------------------------------------------ outputs

    def mouth_open(self) -> float:
        return self.mouth

    def eye_open(self) -> float:
        """0..1 lid opening (blinks squash it briefly)."""
        if self._blink_left <= 0.0:
            return 1.0
        phase = math.sin(math.pi * min(1.0, max(0.0, self._blink_left / 0.15)))
        return max(0.08, 1.0 - 0.92 * phase)

    def breath(self) -> float:
        """Gentle sinusoidal body motion."""
        return math.sin(self.t * 1.8)

    def sway(self) -> float:
        """Slow side-to-side sway, also drives hair swing."""
        return math.sin(self.t * 1.1)

    def look(self) -> tuple[float, float]:
        """Gaze offset (-1..1, x right / y down) per state."""
        table = {
            "LISTENING": (0.0, 0.25),          # attentive, slightly down
            "TRANSCRIBING": (0.35, -0.4),
            "THINKING": (0.5, -0.55),          # eyes up-side "thinking"
            "EXECUTING_TOOL": (0.3, 0.0),
            "ERROR": (0.0, 0.1),
        }
        return table.get(self.state, (0.0, 0.0))

    def brow_raise(self) -> float:
        """>0 surprised/thinking, <0 worried."""
        return {"THINKING": 0.5, "EXECUTING_TOOL": 0.7,
                "ERROR": -0.6}.get(self.state, 0.0)


# ---------------------------------------------------------------- painting

OUTLINE = QColor(40, 30, 60, 110)


def _hair_back_path(style: str, head: QPointF, head_r: float,
                    h: float, s: float) -> QPainterPath:
    """Back-of-the-hair silhouette per hairstyle."""
    back = QPainterPath()
    back.addEllipse(head, head_r * 1.14, head_r * 1.08)
    if style == "long":
        back.addRoundedRect(QRectF(head.x() - head_r * 1.02,
                                   head.y(), head_r * 2.04, head_r * 2.1),
                            head_r * 0.55, head_r * 0.55)
    elif style == "bob":
        back.addRoundedRect(QRectF(head.x() - head_r * 0.98,
                                   head.y() - head_r * 0.1,
                                   head_r * 1.96, head_r * 1.15),
                            head_r * 0.45, head_r * 0.45)
    else:  # twin_tails / ponytail keep the classic shoulder drape
        back.addRoundedRect(QRectF(head.x() - head_r * 0.95,
                                   head.y(), head_r * 1.9, head_r * 1.7),
                            head_r * 0.5, head_r * 0.5)
    return back


def _draw_side_tails(painter: QPainter, style: str, preset: dict,
                     head: QPointF, head_r: float, s: float,
                     sway: float, breath: float) -> None:
    """Twin tails (both sides) or a single side ponytail, drawn behind
    everything; they swing with the sway."""
    sides = (-1.0, 1.0) if style == "twin_tails" else (1.0,)
    length = head_r * (1.55 if style == "twin_tails" else 2.1)
    width = head_r * (0.40 if style == "twin_tails" else 0.52)
    for side in sides:
        anchor = QPointF(head.x() + side * head_r * 0.92,
                         head.y() - head_r * 0.12)
        painter.save()
        painter.translate(anchor)
        painter.rotate(side * (sway * 6.0 + breath * 2.0))
        grad = QLinearGradient(0, 0, 0, length)
        grad.setColorAt(0.0, QColor(preset["hair_light"]))
        grad.setColorAt(1.0, QColor(preset["hair_dark"]))
        path = QPainterPath()
        path.addRoundedRect(QRectF(-width / 2, 0.0, width, length),
                            width / 2, width / 2)
        painter.fillPath(path, QBrush(grad))
        # tie ribbon
        ribbon = QColor(preset["dress_bot"]).lighter(130)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(ribbon)
        painter.drawEllipse(QPointF(0, head_r * 0.04),
                            width * 0.32, width * 0.22)
        painter.restore()


def _draw_dress(painter: QPainter, preset: dict, cx: float,
                body_top: float, body_h: float,
                half_top: float, half_bot: float, s: float) -> None:
    grad = QLinearGradient(cx, body_top, cx, body_top + body_h)
    grad.setColorAt(0.0, QColor(preset["dress_top"]))
    grad.setColorAt(1.0, QColor(preset["dress_bot"]))
    dress = QPainterPath()
    dress.moveTo(cx - half_top, body_top)
    dress.lineTo(cx + half_top, body_top)
    dress.quadTo(cx + half_bot + 6.0 * s, body_top + body_h * 0.65,
                 cx + half_bot, body_top + body_h)
    dress.lineTo(cx - half_bot, body_top + body_h)
    dress.quadTo(cx - half_bot - 6.0 * s, body_top + body_h * 0.65,
                 cx - half_top, body_top)
    dress.closeSubpath()
    painter.fillPath(dress, QBrush(grad))
    painter.setPen(QPen(OUTLINE, 1.2))
    painter.drawPath(dress)
    # frill hem: a row of scallops along the bottom edge
    hem_y = body_top + body_h - 2.0 * s
    n = 6
    step = (half_bot * 2) / n
    frill = QColor(preset["dress_bot"]).lighter(150)
    frill.setAlpha(200)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(frill)
    for i in range(n):
        fx = cx - half_bot + step * (i + 0.5)
        painter.drawEllipse(QPointF(fx, hem_y), step * 0.55, 3.2 * s)


def _draw_eye(painter: QPainter, ex: float, ey: float,
              rx: float, ry: float, eye_color: str,
              openness: float, look_x: float, side: float,
              s: float, lash_color: str) -> None:
    """One large anime eye: gradient iris, pupil, dual highlights,
    lash line with outer spike, and a soft lower lid."""
    if openness <= 0.12:
        # closed/blinking: a gentle downward curve
        pen = QPen(QColor(lash_color), 2.0 * s)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QPointF(ex - rx, ey), QPointF(ex + rx, ey))
        return
    # sclera hint
    painter.setPen(Qt.PenStyle.NoPen)
    sclera = QColor("#ffffff")
    sclera.setAlpha(230)
    painter.setBrush(sclera)
    painter.drawEllipse(QPointF(ex, ey), rx * 1.18, ry * 1.06)
    # iris with vertical gradient (light top -> dark bottom)
    iris_top = QColor(eye_color).lighter(135)
    iris_bot = QColor(eye_color).darker(160)
    g = QLinearGradient(ex, ey - ry, ex, ey + ry)
    g.setColorAt(0.0, iris_top)
    g.setColorAt(1.0, iris_bot)
    painter.setBrush(QBrush(g))
    painter.drawEllipse(QPointF(ex + look_x * rx * 0.25, ey), rx, ry)
    # pupil
    painter.setBrush(QColor("#1a1424"))
    painter.drawEllipse(QPointF(ex + look_x * rx * 0.25, ey), rx * 0.42, ry * 0.46)
    # highlights: big top-left, small bottom-right
    painter.setBrush(QColor("#ffffff"))
    painter.drawEllipse(QPointF(ex - rx * 0.34, ey - ry * 0.38),
                        rx * 0.30, ry * 0.26)
    hl2 = QColor("#ffffff")
    hl2.setAlpha(190)
    painter.setBrush(hl2)
    painter.drawEllipse(QPointF(ex + rx * 0.30, ey + ry * 0.36),
                        rx * 0.16, ry * 0.14)
    # upper lash line with an outer spike
    pen = QPen(QColor(lash_color), 2.4 * s)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    lid_y = ey - ry * 1.06
    painter.drawLine(QPointF(ex - rx * 1.05, lid_y),
                     QPointF(ex + rx * 1.05, lid_y))
    spike_dir = -side   # spike flicks outward
    painter.drawLine(QPointF(ex + spike_dir * rx * 1.0, lid_y),
                     QPointF(ex + spike_dir * rx * 1.35, lid_y - 3.2 * s))
    # lower lid hint
    lower = QColor(lash_color)
    lower.setAlpha(90)
    pen = QPen(lower, 1.2 * s)
    painter.setPen(pen)
    painter.drawLine(QPointF(ex - rx * 0.7, ey + ry * 1.05),
                     QPointF(ex + rx * 0.7, ey + ry * 1.05))


def paint_character(painter: QPainter, a: CharacterAnimator,
                    preset: dict, w: int, h: int) -> None:
    """Paint one frame of the character described by ``preset``."""
    from ui.styles import STATE_COLORS

    style = preset.get("style", "twin_tails")
    s = w / CHAR_W
    breath = a.breath()
    sway = a.sway()
    cx = w / 2 + sway * 2.0 * s

    accent_hex = STATE_COLORS.get(a.state, STATE_COLORS["IDLE"])[0]
    accent = QColor(accent_hex)

    head_r = 34.0 * s
    head = QPointF(cx, 58.0 * s + breath * 2.0 * s)

    # ---------------------------------------------------------- back hair
    painter.fillPath(_hair_back_path(style, head, head_r, h, s),
                     QColor(preset["hair_dark"]))
    _draw_side_tails(painter, style, preset, head, head_r, s, sway, breath)

    # ------------------------------------------------------------ neck
    neck_w = head_r * 0.30
    neck = QPainterPath()
    neck.addRoundedRect(QRectF(head.x() - neck_w / 2,
                               head.y() + head_r * 0.72,
                               neck_w, head_r * 0.55),
                        2.0 * s, 2.0 * s)
    painter.fillPath(neck, QColor(SKIN_SHADE))

    # ------------------------------------------------------------ body
    body_top = 92.0 * s
    body_h = h - body_top - 8.0 * s + breath * 2.0 * s
    half_top, half_bot = 23.0 * s, 37.0 * s
    _draw_dress(painter, preset, cx, body_top, body_h, half_top, half_bot, s)

    # soft shadow under the chin
    shadow = QPainterPath()
    shadow.addEllipse(QPointF(cx, body_top + 4.0 * s),
                      head_r * 0.55, 5.0 * s)
    shade = QColor("#000000")
    shade.setAlpha(45)
    painter.fillPath(shadow, shade)

    # white collar
    collar_w = half_top * 1.5
    collar = QPainterPath()
    collar.addEllipse(QPointF(cx, body_top + 1.0 * s),
                      collar_w / 2, 6.0 * s)
    painter.fillPath(collar, QColor("#efeaff"))

    # arms (behind the head, in front of the dress)
    shoulder_y = body_top + 12.0 * s
    hand_l = QPointF(a.hands[0][0] * w, a.hands[0][1] * h)
    hand_r = QPointF(a.hands[1][0] * w, a.hands[1][1] * h)
    sleeve_pen = QPen(QColor(preset["dress_top"]), 9.0 * s)
    sleeve_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(Qt.PenStyle.NoPen)
    for shoulder_x, hand in ((cx - half_top + 4.0 * s, hand_l),
                             (cx + half_top - 4.0 * s, hand_r)):
        painter.save()
        painter.setPen(sleeve_pen)
        painter.drawLine(QPointF(shoulder_x, shoulder_y), hand)
        painter.restore()
        palm = QPainterPath()
        palm.addEllipse(hand, 6.5 * s, 6.5 * s)
        painter.fillPath(palm, QColor(SKIN))

    # bow under the collar (accent colored = agent state)
    bow_y = body_top + 8.0 * s
    for side in (-1.0, 1.0):
        wing = QPainterPath()
        wing.moveTo(cx, bow_y)
        wing.lineTo(cx + side * 9.0 * s, bow_y - 5.0 * s)
        wing.lineTo(cx + side * 9.0 * s, bow_y + 5.0 * s)
        wing.closeSubpath()
        painter.fillPath(wing, accent)
    knot = QPainterPath()
    knot.addEllipse(QPointF(cx, bow_y), 3.2 * s, 3.2 * s)
    painter.fillPath(knot, accent.lighter(140))

    # ------------------------------------------------------------ head
    skin = QPainterPath()
    skin.addEllipse(head, head_r, head_r * 0.98)
    painter.fillPath(skin, QColor(SKIN))
    painter.setPen(QPen(OUTLINE, 1.2))
    painter.drawPath(skin)

    # ears peeking out
    for side in (-1.0, 1.0):
        ear = QPainterPath()
        ear.addEllipse(QPointF(head.x() + side * head_r * 0.97,
                               head.y() + head_r * 0.15),
                       4.5 * s, 6.5 * s)
        painter.fillPath(ear, QColor(SKIN_SHADE))

    # face shading on the lower cheeks
    cheek_shade = QColor(SKIN_SHADE)
    cheek_shade.setAlpha(90)
    cs = QPainterPath()
    cs.addEllipse(QPointF(head.x(), head.y() + head_r * 0.55),
                  head_r * 0.62, head_r * 0.30)
    painter.fillPath(cs, cheek_shade)

    # ---- bangs: hair cap clipped to the head, scalloped fringe
    painter.save()
    clip = QPainterPath()
    clip.addEllipse(head, head_r * 1.02, head_r * 1.0)
    painter.setClipPath(clip)
    cap = QPainterPath()
    cap.addRect(QRectF(head.x() - head_r * 1.2, head.y() - head_r * 1.2,
                       head_r * 2.4, head_r * 0.95))
    fringe_y = head.y() - head_r * 0.25
    painter.fillPath(cap, QColor(preset["hair_light"]))
    for i, frac in enumerate((-0.62, -0.21, 0.21, 0.62)):
        bump = QPainterPath()
        bump.addEllipse(
            QPointF(head.x() + frac * head_r,
                    fringe_y + head_r * (0.16 if i % 2 == 0 else 0.02)),
            head_r * 0.26, head_r * 0.26)
        painter.fillPath(bump, QColor(preset["hair_light"]))
    # glossy highlight band across the crown
    shine = QColor("#ffffff")
    shine.setAlpha(70)
    shine_pen = QPen(shine, 2.6 * s)
    shine_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(shine_pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    arc = QPainterPath()
    arc.moveTo(head.x() - head_r * 0.55, head.y() - head_r * 0.72)
    arc.quadTo(head.x(), head.y() - head_r * 0.95,
               head.x() + head_r * 0.55, head.y() - head_r * 0.72)
    painter.drawPath(arc)
    painter.restore()

    # ahoge (single strand on top) — sways gently
    painter.save()
    painter.translate(head.x(), head.y() - head_r * 0.95)
    painter.rotate(-sway * 10.0)
    ahoge = QPainterPath()
    ahoge.moveTo(0.0, 0.0)
    ahoge.cubicTo(6.0 * s, -12.0 * s, 16.0 * s, -10.0 * s, 20.0 * s, -2.0 * s)
    painter.setPen(QPen(QColor(preset["hair_light"]), 2.6 * s,
                        Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(ahoge)
    painter.restore()

    # ribbons where the tails attach
    if style in ("twin_tails", "ponytail"):
        sides = (-1.0, 1.0) if style == "twin_tails" else (1.0,)
        for side in sides:
            ax = head.x() + side * head_r * 0.88
            ay = head.y() - head_r * 0.12
            for dy in (-1.0, 1.0):
                tri = QPainterPath()
                tri.moveTo(ax, ay)
                tri.lineTo(ax + side * 8.0 * s, ay + dy * 5.0 * s)
                tri.lineTo(ax + side * 3.0 * s, ay + dy * 2.0 * s)
                tri.closeSubpath()
                painter.fillPath(tri, accent)
            center = QPainterPath()
            center.addEllipse(QPointF(ax, ay), 3.0 * s, 3.0 * s)
            painter.fillPath(center, accent.lighter(130))

    # ---- face
    look_x, look_y = a.look()
    openness = a.eye_open()
    eye_dx = head_r * 0.42
    eye_dy = head_r * 0.10
    lash = preset.get("lash", preset["hair_dark"])
    for side in (-1.0, 1.0):
        ex = head.x() + side * eye_dx + look_x * head_r * 0.09
        ey = head.y() + eye_dy + look_y * head_r * 0.12
        _draw_eye(painter, ex, ey,
                  head_r * 0.145, head_r * 0.30 * openness,
                  preset["eye"], openness, look_x, side, s, lash)

    # brows
    brow_raise = a.brow_raise()
    brow_y = head.y() - head_r * 0.20 - brow_raise * head_r * 0.10
    tilt = 0.16 if a.state == "ERROR" else 0.03
    brow_pen = QPen(QColor(lash), 2.0 * s)
    brow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(brow_pen)
    for side in (-1.0, 1.0):
        bx = head.x() + side * eye_dx
        painter.drawLine(
            QPointF(bx - head_r * 0.13, brow_y + side * tilt * head_r * 0.22),
            QPointF(bx + head_r * 0.13, brow_y))

    # blush
    blush_a = 52 if a.state != "IDLE" else 32
    for side in (-1.0, 1.0):
        bl = QPainterPath()
        bl.addEllipse(QPointF(head.x() + side * head_r * 0.58,
                              head.y() + head_r * 0.34),
                      head_r * 0.13, head_r * 0.075)
        color = QColor("#ef8b8b")
        color.setAlpha(blush_a)
        painter.fillPath(bl, color)

    # ---- mouth (driven by live audio amplitude while SPEAKING)
    mo = a.mouth_open()
    mouth_y = head.y() + head_r * 0.48
    mw = head_r * (0.20 + 0.10 * mo)
    mh = max(1.2 * s, head_r * (0.04 + 0.34 * mo))
    if mo < 0.08:
        smile = QPainterPath()
        smile.moveTo(head.x() - mw / 2, mouth_y)
        smile.quadTo(head.x(), mouth_y + head_r * 0.10,
                     head.x() + mw / 2, mouth_y)
        painter.setPen(QPen(QColor("#8a4a5c"), 2.0 * s,
                            Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawPath(smile)
    else:
        mp = QPainterPath()
        mp.addRoundedRect(QRectF(head.x() - mw / 2, mouth_y - mh / 2,
                                 mw, mh), mh / 2, mh / 2)
        painter.fillPath(mp, QColor("#5b2733"))
        tongue = QPainterPath()
        tongue.addEllipse(QPointF(head.x(), mouth_y + mh * 0.28),
                          mw * 0.30, mh * 0.30)
        tcol = QColor("#d96a80")
        painter.fillPath(tongue, tcol)

    # ---- state diamond floating above the head
    pulse = 0.5 + 0.5 * math.sin(a.t * 3.0)
    size = (5.0 + 1.5 * pulse) * s
    dy = head.y() - head_r * 1.32 - pulse * 2.0 * s
    diamond = QPainterPath()
    diamond.moveTo(cx, dy - size)
    diamond.lineTo(cx + size, dy)
    diamond.lineTo(cx, dy + size)
    diamond.lineTo(cx - size, dy)
    diamond.closeSubpath()
    glow = QColor(accent_hex)
    glow.setAlpha(90)
    halo = QPainterPath()
    halo.addEllipse(QPointF(cx, dy), size * 1.8, size * 1.8)
    painter.fillPath(halo, glow)
    painter.fillPath(diamond, accent)


def render_preview(preset_id: str, size: int = 120) -> "QImage":
    """Render one static frame of a preset (IDLE pose) into a transparent
    QImage — used by the Settings dialog for character thumbnails."""
    image = QImage(size, int(size * CHAR_H / CHAR_W),
                   QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        animator = CharacterAnimator()
        animator.tick(0.033)
        paint_character(painter, animator,
                        CHARACTER_PRESETS.get(preset_id,
                                              CHARACTER_PRESETS[DEFAULT_CHARACTER]),
                        image.width(), image.height())
    finally:
        painter.end()
    return image


class CharacterWidget(QWidget):
    """Frameless always-on-top character window that follows the island."""

    def __init__(self, settings=None, island=None) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool)
        self.settings = settings
        self.island = island
        self.animator = CharacterAnimator()
        self.preset_id = DEFAULT_CHARACTER
        if settings is not None:
            saved = str(settings.get("character.preset", DEFAULT_CHARACTER))
            if saved in CHARACTER_PRESETS:
                self.preset_id = saved

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # Purely decorative: it must never steal hover/clicks — especially
        # from the island's gear/close buttons it may visually overlap.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.apply_scale()

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ------------------------------------------------------------- config

    def apply_scale(self) -> None:
        scale = self._scale()
        self.setFixedSize(int(CHAR_W * scale), int(CHAR_H * scale))

    def _scale(self) -> float:
        try:
            value = self.settings.get("character.scale", 1.0)
            return max(0.5, min(2.0, float(value)))
        except Exception:
            return 1.0

    def _offset(self) -> tuple[int, int]:
        try:
            ox, oy = self.settings.get("character.offset", [-30, -190])
            s = self._scale()
            return int(ox * s), int(oy * s)
        except Exception:
            return -30, -190

    def follow_island(self, island_pos) -> None:
        ox, oy = self._offset()
        self.move(island_pos.x() + ox, island_pos.y() + oy)

    # --------------------------------------------------------------- state

    @property
    def preset(self) -> dict:
        return CHARACTER_PRESETS.get(self.preset_id,
                                     CHARACTER_PRESETS[DEFAULT_CHARACTER])

    def set_preset(self, preset_id: str) -> bool:
        """Switch character (live). Returns True when the id was known."""
        if preset_id not in CHARACTER_PRESETS:
            return False
        self.preset_id = preset_id
        self.update()
        return True

    def set_state(self, state: str) -> None:
        self.animator.set_state(state)

    def set_mouth_level(self, level: float) -> None:
        self.animator.set_mouth_level(level)

    # ----------------------------------------------------------- animation

    def _tick(self) -> None:
        self.animator.tick(0.033)
        self.update()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.island is not None:
            self.follow_island(self.island.pos())

    # ---------------------------------------------------------------- paint

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
            paint_character(painter, self.animator, self.preset,
                            self.width(), self.height())
        finally:
            painter.end()
