"""
Hand DJ Controller — Cursor Mode
==================================
Two cursors (one per hand) controlled by fingertip position.
Pinch to grab whatever the cursor is over, then:
  - Rotate wrist → spin jogwheel or turn knob
  - Move hand up/down → slide fader
  - Release pinch → let go

Much more precise than palm-based tracking since only the
fingertip position matters for targeting.

Requirements:
  pip install mediapipe opencv-python pygame numpy

Usage:
  python hand_dj_cursor.py
  python hand_dj_cursor.py --camera 1
  python hand_dj_cursor.py --flip-camera
"""

import cv2
import mediapipe as mp
import pygame
import numpy as np
import math
import sys
import argparse
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

# ─── Configuration ───────────────────────────────────────────────────────────

PROJ_WIDTH = 854
PROJ_HEIGHT = 480
DEBUG_WIDTH = 1280
DEBUG_HEIGHT = 600
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS = 60

# Colors
BG_COLOR = (12, 12, 18)
ACCENT_CYAN = (0, 220, 255)
ACCENT_MAGENTA = (255, 0, 180)
ACCENT_WHITE = (230, 230, 240)
ACCENT_DIM = (60, 60, 80)
KNOB_BG = (35, 35, 50)
KNOB_RING = (50, 50, 70)
TEXT_DIM = (100, 100, 130)
JOGWHEEL_BG = (18, 18, 28)
INDICATOR_GREEN = (0, 255, 120)
INDICATOR_RED = (255, 50, 80)
CURSOR_A_COLOR = (0, 255, 200)
CURSOR_B_COLOR = (255, 80, 220)

# Interaction
PINCH_THRESHOLD = 0.07        # Thumb-to-index distance
GRAB_RADIUS = 40              # How close cursor must be to grab a control
CURSOR_SMOOTH = 0.3           # Cursor position smoothing (0=instant, 1=frozen)
ROTATION_SMOOTH = 0.3         # Rotation smoothing


# ─── DJ Controls ─────────────────────────────────────────────────────────────

@dataclass
class JogWheel:
    cx: int = 0
    cy: int = 0
    radius: int = 90
    angle: float = 0.0
    velocity: float = 0.0
    grabbed: bool = False
    last_rotation: float = 0.0
    deck_name: str = "A"
    color: Tuple[int, int, int] = (0, 220, 255)

    def update(self, dt: float):
        if not self.grabbed:
            self.velocity *= 0.96
            self.angle += self.velocity * dt
        self.angle %= 360

    def get_rpm(self) -> float:
        return abs(self.velocity) / 6.0


@dataclass
class Knob:
    cx: int = 0
    cy: int = 0
    radius: int = 22
    value: float = 0.5
    grabbed: bool = False
    last_rotation: float = 0.0
    label: str = "FILTER"
    color: Tuple[int, int, int] = (0, 220, 255)
    smoothed_value: float = 0.5

    def update(self):
        self.smoothed_value += (self.value - self.smoothed_value) * 0.2


@dataclass
class Fader:
    cx: int = 0
    cy: int = 0
    width: int = 20
    height: int = 120
    value: float = 0.5
    grabbed: bool = False
    last_y: float = 0.0
    last_x: float = 0.0
    label: str = "VOLUME"
    color: Tuple[int, int, int] = (0, 220, 255)
    smoothed_value: float = 0.5
    horizontal: bool = False

    def update(self):
        self.smoothed_value += (self.value - self.smoothed_value) * 0.2


# ─── Hand Cursor ─────────────────────────────────────────────────────────────

@dataclass
class HandCursor:
    """Represents one hand's cursor on the DJ board."""
    # Smoothed position on the board (pixels)
    x: float = 0.0
    y: float = 0.0
    # Raw normalized position from camera
    raw_x: float = 0.0
    raw_y: float = 0.0
    # Hand rotation (degrees)
    rotation: float = 0.0
    smoothed_rotation: float = 0.0
    # Pinch state
    pinching: bool = False
    # What control is currently grabbed (None if nothing)
    grabbed_control: object = None
    # Is this hand detected?
    active: bool = False
    # Which hand
    side: str = "L"
    color: Tuple[int, int, int] = (0, 255, 200)


# ─── Hand Tracker ────────────────────────────────────────────────────────────

class HandTracker:
    """Minimal hand tracker — extracts cursor position, rotation, and pinch."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def extract_hand_data(self, hand_lm) -> dict:
        """Extract the 4 things we care about from a hand."""
        # Cursor position = index fingertip
        idx_tip = hand_lm.landmark[8]
        cursor_x = idx_tip.x
        cursor_y = idx_tip.y

        # Rotation = angle from wrist to middle fingertip
        wrist = hand_lm.landmark[0]
        mid_tip = hand_lm.landmark[12]
        rotation = math.degrees(math.atan2(
            mid_tip.y - wrist.y, mid_tip.x - wrist.x))

        # Pinch = thumb tip to index tip distance
        thumb_tip = hand_lm.landmark[4]
        pinch_dist = math.sqrt(
            (thumb_tip.x - idx_tip.x)**2 + (thumb_tip.y - idx_tip.y)**2)
        pinching = pinch_dist < PINCH_THRESHOLD

        # Vertical position (for faders) = index tip Y
        vert_y = cursor_y

        return {
            'cursor_x': cursor_x,
            'cursor_y': cursor_y,
            'rotation': rotation,
            'pinching': pinching,
            'vert_y': vert_y,
        }

    def draw_landmarks(self, frame, results):
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hl, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 200), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(0, 150, 255), thickness=1),
                )
        return frame


# ─── Board Renderer ──────────────────────────────────────────────────────────

class BoardRenderer:
    """Renders the DJ board onto a surface."""

    def __init__(self, surface: pygame.Surface):
        self.s = surface
        self.font_label = pygame.font.SysFont("Helvetica", 14, bold=True)
        self.font_small = pygame.font.SysFont("Helvetica", 11)
        self.font_tiny = pygame.font.SysFont("Helvetica", 9)

    def clear(self):
        self.s.fill(BG_COLOR)

    def draw_jogwheel(self, w: JogWheel):
        cx, cy, r = w.cx, w.cy, w.radius

        # Glow when grabbed
        if w.grabbed:
            for i in range(4):
                c = tuple(min(255, v // (i + 1)) for v in w.color)
                pygame.draw.circle(self.s, c, (cx, cy), r + 10 - i * 3, 2)

        pygame.draw.circle(self.s, JOGWHEEL_BG, (cx, cy), r)
        pygame.draw.circle(self.s, KNOB_RING, (cx, cy), r, 2)

        # Platter lines
        for i in range(12):
            a = math.radians(w.angle + i * 30)
            x1 = cx + math.cos(a) * (r * 0.25)
            y1 = cy + math.sin(a) * (r * 0.25)
            x2 = cx + math.cos(a) * (r * 0.85)
            y2 = cy + math.sin(a) * (r * 0.85)
            color = w.color if i % 3 == 0 else ACCENT_DIM
            pygame.draw.line(self.s, color, (x1, y1), (x2, y2), 2 if i % 3 == 0 else 1)

        # Center
        pygame.draw.circle(self.s, w.color, (cx, cy), 6)
        pygame.draw.circle(self.s, BG_COLOR, (cx, cy), 3)

        # Position indicator
        ia = math.radians(w.angle)
        ix = cx + math.cos(ia) * (r * 0.92)
        iy = cy + math.sin(ia) * (r * 0.92)
        pygame.draw.circle(self.s, w.color, (int(ix), int(iy)), 4)

        # Label
        lbl = self.font_label.render(f"DECK {w.deck_name}", True, w.color)
        self.s.blit(lbl, (cx - lbl.get_width() // 2, cy - r - 22))

        # Grab zone (faint circle)
        for a in range(0, 360, 6):
            angle = math.radians(a)
            zx = cx + math.cos(angle) * GRAB_RADIUS
            zy = cy + math.sin(angle) * GRAB_RADIUS
            # Don't draw — zone is the control itself, not a separate ring

    def draw_knob(self, k: Knob):
        cx, cy, r = k.cx, k.cy, k.radius

        # Glow when grabbed
        if k.grabbed:
            pygame.draw.circle(self.s, k.color, (cx, cy), r + 6, 2)

        pygame.draw.circle(self.s, KNOB_BG, (cx, cy), r)

        # Background arc
        for a in range(0, 270, 3):
            angle = math.radians(135 + a)
            x = cx + math.cos(angle) * (r - 3)
            y = cy + math.sin(angle) * (r - 3)
            pygame.draw.circle(self.s, KNOB_RING, (int(x), int(y)), 1)

        # Value arc
        for a in range(0, int(270 * k.smoothed_value), 3):
            angle = math.radians(135 + a)
            x = cx + math.cos(angle) * (r - 3)
            y = cy + math.sin(angle) * (r - 3)
            pygame.draw.circle(self.s, k.color, (int(x), int(y)), 2)

        # Pointer
        pa = math.radians(135 + 270 * k.smoothed_value)
        px = cx + math.cos(pa) * (r * 0.55)
        py = cy + math.sin(pa) * (r * 0.55)
        pygame.draw.line(self.s, ACCENT_WHITE, (cx, cy), (int(px), int(py)), 2)

        pygame.draw.circle(self.s, k.color if k.grabbed else ACCENT_DIM, (cx, cy), 3)

        lbl = self.font_tiny.render(k.label, True, k.color)
        self.s.blit(lbl, (cx - lbl.get_width() // 2, cy + r + 5))

    def draw_fader(self, f: Fader):
        cx, cy = f.cx, f.cy
        w, h = f.width, f.height

        if f.horizontal:
            # Horizontal (crossfader)
            track = (cx - w // 2, cy - h // 2, w, h)
            pygame.draw.rect(self.s, KNOB_BG, track, border_radius=4)
            pygame.draw.rect(self.s, KNOB_RING, track, 1, border_radius=4)
            thumb_x = cx - w // 2 + int(w * f.smoothed_value)
            pygame.draw.rect(self.s, ACCENT_WHITE if f.grabbed else ACCENT_DIM,
                             (thumb_x - 6, cy - h // 2 - 2, 12, h + 4), border_radius=3)
            # A/B labels
            a_lbl = self.font_tiny.render("A", True, ACCENT_CYAN)
            b_lbl = self.font_tiny.render("B", True, ACCENT_MAGENTA)
            self.s.blit(a_lbl, (cx - w // 2 - 14, cy - 5))
            self.s.blit(b_lbl, (cx + w // 2 + 6, cy - 5))
        else:
            # Vertical (volume)
            track = (cx - w // 2, cy - h // 2, w, h)
            pygame.draw.rect(self.s, KNOB_BG, track, border_radius=4)
            pygame.draw.rect(self.s, KNOB_RING, track, 1, border_radius=4)
            fill_h = int(h * f.smoothed_value)
            if fill_h > 0:
                fill_rect = (cx - w // 2 + 2, cy + h // 2 - fill_h, w - 4, fill_h)
                pygame.draw.rect(self.s, f.color, fill_rect, border_radius=2)
            thumb_y = cy + h // 2 - int(h * f.smoothed_value)
            pygame.draw.rect(self.s, ACCENT_WHITE if f.grabbed else ACCENT_DIM,
                             (cx - w // 2 - 2, thumb_y - 4, w + 4, 8), border_radius=3)

        # Glow when grabbed
        if f.grabbed:
            r = max(w, h) // 2 + 8
            pygame.draw.rect(self.s, f.color,
                             (cx - w // 2 - 4, cy - h // 2 - 4, w + 8, h + 8), 1, border_radius=6)

        lbl = self.font_tiny.render(f.label, True, f.color)
        self.s.blit(lbl, (cx - lbl.get_width() // 2, cy + h // 2 + 8))

    def draw_cursor(self, cursor: HandCursor):
        """Draw a cursor on the board."""
        if not cursor.active:
            return

        x, y = int(cursor.x), int(cursor.y)
        color = cursor.color

        if cursor.pinching:
            # Pinching = solid dot + ring
            pygame.draw.circle(self.s, color, (x, y), 10, 2)
            pygame.draw.circle(self.s, color, (x, y), 4)
            # Show rotation indicator
            ra = math.radians(cursor.smoothed_rotation)
            rx = x + math.cos(ra) * 16
            ry = y + math.sin(ra) * 16
            pygame.draw.line(self.s, color, (x, y), (int(rx), int(ry)), 2)
        else:
            # Hovering = crosshair
            pygame.draw.circle(self.s, color, (x, y), 6, 1)
            pygame.draw.line(self.s, color, (x - 10, y), (x + 10, y), 1)
            pygame.draw.line(self.s, color, (x, y - 10), (x, y + 10), 1)

        # Hand label
        lbl = self.font_tiny.render(cursor.side, True, color)
        self.s.blit(lbl, (x + 12, y - 6))

        # Show what it's grabbing
        if cursor.grabbed_control is not None:
            ctrl = cursor.grabbed_control
            name = getattr(ctrl, 'label', None) or getattr(ctrl, 'deck_name', '?')
            grab_lbl = self.font_tiny.render(f"→ {name}", True, color)
            self.s.blit(grab_lbl, (x + 12, y + 6))


# ─── Debug Panel ─────────────────────────────────────────────────────────────

class DebugPanel:
    """Green-on-black debug panel for the right side."""

    LOG_BG = (10, 14, 10)
    LOG_GREEN = (0, 255, 80)
    LOG_DIM = (0, 140, 50)
    LOG_BRIGHT = (80, 255, 140)
    LOG_GRABBED = (255, 255, 80)

    def __init__(self, surface: pygame.Surface, x_offset: int):
        self.s = surface
        self.x0 = x_offset
        self.font = pygame.font.SysFont("Courier", 13)
        self.font_bold = pygame.font.SysFont("Courier", 13, bold=True)
        self.font_small = pygame.font.SysFont("Courier", 11)
        self.font_header = pygame.font.SysFont("Courier", 10, bold=True)
        self.font_title = pygame.font.SysFont("Helvetica", 18, bold=True)
        self.log_lines: List[str] = []
        self.max_log = 12

    def add_event(self, msg: str):
        import time
        ts = time.strftime("%H:%M:%S")
        self.log_lines.append(f"[{ts}] {msg}")
        if len(self.log_lines) > self.max_log:
            self.log_lines.pop(0)

    def draw(self, cursors, wheels, knobs, faders, fps):
        px = self.x0
        pw = DEBUG_WIDTH - px

        # Background
        pygame.draw.rect(self.s, self.LOG_BG, (px, 0, pw, DEBUG_HEIGHT))
        pygame.draw.line(self.s, (0, 80, 40), (px, 0), (px, DEBUG_HEIGHT), 2)

        x = px + 12
        y = 10

        title = self.font_title.render("DEBUG LOG", True, self.LOG_GREEN)
        self.s.blit(title, (x, y))
        fps_txt = self.font_small.render(f"{fps:.0f} FPS", True, self.LOG_DIM)
        self.s.blit(fps_txt, (px + pw - 65, y + 4))
        y += 28

        pygame.draw.line(self.s, (0, 60, 30), (px + 8, y), (px + pw - 8, y), 1)
        y += 8

        # Cursors
        self.s.blit(self.font_header.render("── CURSORS ──", True, self.LOG_DIM), (x, y))
        y += 16
        for c in cursors:
            if c.active:
                grab_name = "---"
                if c.grabbed_control:
                    grab_name = getattr(c.grabbed_control, 'label', None) or \
                                f"DECK {getattr(c.grabbed_control, 'deck_name', '?')}"
                pinch_str = "PINCH" if c.pinching else "hover"
                color = self.LOG_GRABBED if c.pinching else self.LOG_BRIGHT
                line = f" {c.side}: ({c.x:4.0f},{c.y:4.0f}) {pinch_str:>5} → {grab_name}"
            else:
                color = self.LOG_DIM
                line = f" {c.side}: not detected"
            self.s.blit(self.font.render(line, True, color), (x, y))
            y += 17
        y += 8

        # Decks
        self.s.blit(self.font_header.render("── DECKS ──", True, self.LOG_DIM), (x, y))
        y += 16
        for w in wheels:
            color = self.LOG_GRABBED if w.grabbed else self.LOG_GREEN
            tag = " ★" if w.grabbed else ""
            line = f" DECK {w.deck_name}: {w.angle:6.1f}° {w.velocity:+7.1f}{tag}"
            self.s.blit(self.font.render(line, True, color), (x, y))
            y += 17
        y += 6

        # Knobs
        self.s.blit(self.font_header.render("── KNOBS ──", True, self.LOG_DIM), (x, y))
        y += 16
        for k in knobs:
            color = self.LOG_GRABBED if k.grabbed else self.LOG_GREEN
            tag = " ★" if k.grabbed else ""
            filled = int(10 * k.smoothed_value)
            bar = "█" * filled + "░" * (10 - filled)
            line = f" {k.label:>10}: {bar} {k.smoothed_value*100:4.0f}%{tag}"
            self.s.blit(self.font.render(line, True, color), (x, y))
            y += 17
        y += 6

        # Faders
        self.s.blit(self.font_header.render("── FADERS ──", True, self.LOG_DIM), (x, y))
        y += 16
        for f in faders:
            color = self.LOG_GRABBED if f.grabbed else self.LOG_GREEN
            tag = " ★" if f.grabbed else ""
            filled = int(10 * f.smoothed_value)
            bar = "█" * filled + "░" * (10 - filled)
            line = f" {f.label:>10}: {bar} {f.smoothed_value*100:4.0f}%{tag}"
            self.s.blit(self.font.render(line, True, color), (x, y))
            y += 17
        y += 10

        # Event log
        pygame.draw.line(self.s, (0, 60, 30), (px + 8, y), (px + pw - 8, y), 1)
        y += 6
        self.s.blit(self.font_header.render("── EVENT LOG ──", True, self.LOG_DIM), (x, y))
        y += 14
        for line in self.log_lines[-8:]:
            self.s.blit(self.font_small.render(line, True, self.LOG_DIM), (x, y))
            y += 14


# ─── Main App ────────────────────────────────────────────────────────────────

class HandDJApp:

    def __init__(self, camera_idx=0, flip_camera=False):
        pygame.init()
        self.screen = pygame.display.set_mode((DEBUG_WIDTH, DEBUG_HEIGHT))
        pygame.display.set_caption("Hand DJ — Cursor Mode")
        self.clock = pygame.time.Clock()
        self.flip_camera = flip_camera

        # Offscreen board surface
        self.board_surface = pygame.Surface((PROJ_WIDTH, PROJ_HEIGHT))
        self.board_renderer = BoardRenderer(self.board_surface)

        # Debug panel
        self.panel_x = 860
        self.debug_panel = DebugPanel(self.screen, self.panel_x)

        # DJ Controls
        self.wheel_a = JogWheel(deck_name="A", color=ACCENT_CYAN)
        self.wheel_b = JogWheel(deck_name="B", color=ACCENT_MAGENTA)
        self.knob_filter_a = Knob(label="FILTER A", color=ACCENT_CYAN)
        self.knob_filter_b = Knob(label="FILTER B", color=ACCENT_MAGENTA)
        self.knob_eq_hi_a = Knob(label="HI A", color=ACCENT_CYAN)
        self.knob_eq_hi_b = Knob(label="HI B", color=ACCENT_MAGENTA)
        self.fader_vol_a = Fader(label="VOL A", color=ACCENT_CYAN)
        self.fader_vol_b = Fader(label="VOL B", color=ACCENT_MAGENTA)
        self.crossfader = Fader(label="CROSSFADER", color=ACCENT_WHITE, horizontal=True)

        self._layout()

        # All controls in a flat list for hit-testing
        self.all_controls = [
            self.wheel_a, self.wheel_b,
            self.knob_filter_a, self.knob_filter_b,
            self.knob_eq_hi_a, self.knob_eq_hi_b,
            self.fader_vol_a, self.fader_vol_b,
            self.crossfader,
        ]

        # Two cursors
        self.cursor_l = HandCursor(side="L", color=CURSOR_A_COLOR)
        self.cursor_r = HandCursor(side="R", color=CURSOR_B_COLOR)

        # Hand tracker
        self.tracker = HandTracker()

        # Camera
        self.cap = cv2.VideoCapture(camera_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        if not self.cap.isOpened():
            print(f"ERROR: Cannot open camera {camera_idx}")
            sys.exit(1)

        self.running = True
        self.last_frame = None

    def _layout(self):
        pw, ph = PROJ_WIDTH, PROJ_HEIGHT

        # Jogwheels
        self.wheel_a.cx, self.wheel_a.cy, self.wheel_a.radius = 150, ph // 2, 90
        self.wheel_b.cx, self.wheel_b.cy, self.wheel_b.radius = pw - 150, ph // 2, 90

        # Knobs above jogwheels
        self.knob_filter_a.cx, self.knob_filter_a.cy = 110, 50
        self.knob_filter_b.cx, self.knob_filter_b.cy = pw - 110, 50
        self.knob_eq_hi_a.cx, self.knob_eq_hi_a.cy = 190, 50
        self.knob_eq_hi_b.cx, self.knob_eq_hi_b.cy = pw - 190, 50

        # Volume faders
        self.fader_vol_a.cx, self.fader_vol_a.cy = 310, ph // 2
        self.fader_vol_a.width, self.fader_vol_a.height = 18, 140
        self.fader_vol_b.cx, self.fader_vol_b.cy = pw - 310, ph // 2
        self.fader_vol_b.width, self.fader_vol_b.height = 18, 140

        # Crossfader
        self.crossfader.cx, self.crossfader.cy = pw // 2, ph - 35
        self.crossfader.width, self.crossfader.height = 220, 14
        self.crossfader.horizontal = True

    def _hit_test(self, cx: float, cy: float) -> object:
        """Find the closest control to the cursor position."""
        best = None
        best_dist = float('inf')

        for ctrl in self.all_controls:
            dx = cx - ctrl.cx
            dy = cy - ctrl.cy
            dist = math.sqrt(dx*dx + dy*dy)

            # Determine hit radius based on control type
            if isinstance(ctrl, JogWheel):
                hit_r = ctrl.radius + 10
            elif isinstance(ctrl, Knob):
                hit_r = ctrl.radius + 15
            elif isinstance(ctrl, Fader):
                # Rectangular hit test
                hw = ctrl.width // 2 + 15
                hh = ctrl.height // 2 + 15
                if ctrl.horizontal:
                    hw = ctrl.width // 2 + 15
                    hh = ctrl.height // 2 + 15
                if abs(dx) < hw and abs(dy) < hh:
                    dist = 0  # Inside fader = perfect hit
                else:
                    continue
            else:
                hit_r = GRAB_RADIUS

            if dist < hit_r and dist < best_dist:
                best = ctrl
                best_dist = dist

        return best

    def _update_cursor(self, cursor: HandCursor, hand_data: Optional[dict]):
        """Update a cursor from hand tracking data."""
        if hand_data is None:
            cursor.active = False
            # Release any grabbed control
            if cursor.grabbed_control is not None:
                cursor.grabbed_control.grabbed = False
                self.debug_panel.add_event(f"DROP  {self._ctrl_name(cursor.grabbed_control)}")
                cursor.grabbed_control = None
            return

        cursor.active = True

        # Smooth cursor position
        target_x = hand_data['cursor_x'] * PROJ_WIDTH
        target_y = hand_data['cursor_y'] * PROJ_HEIGHT
        cursor.x = cursor.x * CURSOR_SMOOTH + target_x * (1 - CURSOR_SMOOTH)
        cursor.y = cursor.y * CURSOR_SMOOTH + target_y * (1 - CURSOR_SMOOTH)

        # Smooth rotation
        new_rot = hand_data['rotation']
        cursor.smoothed_rotation = cursor.smoothed_rotation * ROTATION_SMOOTH + \
                                   new_rot * (1 - ROTATION_SMOOTH)

        was_pinching = cursor.pinching
        cursor.pinching = hand_data['pinching']

        # ── Grab / Release logic ──
        if cursor.pinching and not was_pinching:
            # Just started pinching — try to grab
            hit = self._hit_test(cursor.x, cursor.y)
            if hit is not None and not hit.grabbed:
                cursor.grabbed_control = hit
                hit.grabbed = True
                # Store initial rotation for delta tracking
                if isinstance(hit, (JogWheel, Knob)):
                    hit.last_rotation = cursor.smoothed_rotation
                elif isinstance(hit, Fader):
                    if hit.horizontal:
                        hit.last_x = hand_data['cursor_x']
                    else:
                        hit.last_y = hand_data['cursor_y']
                self.debug_panel.add_event(f"GRAB  {self._ctrl_name(hit)}")

        elif not cursor.pinching and was_pinching:
            # Just released pinch — drop
            if cursor.grabbed_control is not None:
                self.debug_panel.add_event(f"DROP  {self._ctrl_name(cursor.grabbed_control)}")
                cursor.grabbed_control.grabbed = False
                cursor.grabbed_control = None

        # ── Control update while grabbed ──
        ctrl = cursor.grabbed_control
        if ctrl is not None and cursor.pinching:
            if isinstance(ctrl, JogWheel):
                # Rotation delta → spin
                delta = cursor.smoothed_rotation - ctrl.last_rotation
                if delta > 180: delta -= 360
                elif delta < -180: delta += 360
                ctrl.angle += delta * 1.5
                ctrl.velocity = delta * FPS
                ctrl.last_rotation = cursor.smoothed_rotation

            elif isinstance(ctrl, Knob):
                # Rotation delta → turn knob
                delta = cursor.smoothed_rotation - ctrl.last_rotation
                if delta > 180: delta -= 360
                elif delta < -180: delta += 360
                ctrl.value = max(0.0, min(1.0, ctrl.value + delta / 270.0))
                ctrl.last_rotation = cursor.smoothed_rotation

            elif isinstance(ctrl, Fader):
                if ctrl.horizontal:
                    # Horizontal movement → crossfader
                    new_x = hand_data['cursor_x']
                    delta = new_x - ctrl.last_x
                    ctrl.value = max(0.0, min(1.0, ctrl.value + delta * 2.5))
                    ctrl.last_x = new_x
                else:
                    # Vertical movement → volume fader
                    new_y = hand_data['cursor_y']
                    delta = ctrl.last_y - new_y  # Inverted: up = increase
                    ctrl.value = max(0.0, min(1.0, ctrl.value + delta * 2.5))
                    ctrl.last_y = new_y

    def _ctrl_name(self, ctrl) -> str:
        return getattr(ctrl, 'label', None) or f"DECK {getattr(ctrl, 'deck_name', '?')}"

    def run(self):
        print("\n" + "=" * 50)
        print("  HAND DJ — CURSOR MODE")
        print("  Move hands to position cursors")
        print("  Pinch to grab, rotate/move to control")
        print("  Release pinch to let go")
        print("  Q / ESC to quit")
        print("=" * 50 + "\n")

        while self.running:
            dt = self.clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        self.running = False

            # Camera
            ret, frame = self.cap.read()
            left_data = None
            right_data = None

            if ret:
                frame = cv2.flip(frame, -1 if self.flip_camera else 1)
                results = self.tracker.process(frame)

                if results.multi_hand_landmarks:
                    for idx, hand_lm in enumerate(results.multi_hand_landmarks):
                        data = self.tracker.extract_hand_data(hand_lm)

                        # Determine handedness
                        if results.multi_handedness and idx < len(results.multi_handedness):
                            label = results.multi_handedness[idx].classification[0].label
                            # MediaPipe labels are mirrored
                            side = "Left" if label == "Left" else "Right"
                        else:
                            side = "Left" if data['cursor_x'] < 0.5 else "Right"

                        if side == "Left":
                            left_data = data
                        else:
                            right_data = data

                frame = self.tracker.draw_landmarks(frame, results) if results else frame
                self.last_frame = frame

            # Update cursors
            self._update_cursor(self.cursor_l, left_data)
            self._update_cursor(self.cursor_r, right_data)

            # Update controls
            self.wheel_a.update(dt)
            self.wheel_b.update(dt)
            for k in [self.knob_filter_a, self.knob_filter_b,
                      self.knob_eq_hi_a, self.knob_eq_hi_b]:
                k.update()
            for f in [self.fader_vol_a, self.fader_vol_b, self.crossfader]:
                f.update()

            # ── Render board ──
            self.board_renderer.clear()
            self.board_renderer.draw_jogwheel(self.wheel_a)
            self.board_renderer.draw_jogwheel(self.wheel_b)
            for k in [self.knob_filter_a, self.knob_filter_b,
                      self.knob_eq_hi_a, self.knob_eq_hi_b]:
                self.board_renderer.draw_knob(k)
            self.board_renderer.draw_fader(self.fader_vol_a)
            self.board_renderer.draw_fader(self.fader_vol_b)
            self.board_renderer.draw_fader(self.crossfader)
            # Cursors on top
            self.board_renderer.draw_cursor(self.cursor_l)
            self.board_renderer.draw_cursor(self.cursor_r)

            # ── Render debug window ──
            self.screen.fill((16, 16, 24))

            # Header
            hdr_font = pygame.font.SysFont("Helvetica", 18, bold=True)
            hdr = hdr_font.render("HAND DJ — CURSOR MODE", True, ACCENT_CYAN)
            self.screen.blit(hdr, (10, 10))
            sub = pygame.font.SysFont("Courier", 11).render(
                "Pinch=Grab  Rotate=Spin/Turn  Move=Slide  |  Q/ESC=Quit", True, TEXT_DIM)
            self.screen.blit(sub, (10, 34))

            # Board preview (left panel)
            scale = min((self.panel_x - 20) / PROJ_WIDTH,
                        (DEBUG_HEIGHT - 60) / PROJ_HEIGHT)
            sw = int(PROJ_WIDTH * scale)
            sh = int(PROJ_HEIGHT * scale)
            scaled = pygame.transform.smoothscale(self.board_surface, (sw, sh))
            pygame.draw.rect(self.screen, (40, 40, 60), (8, 48, sw + 4, sh + 4), 2)
            self.screen.blit(scaled, (10, 50))

            # Debug panel (right)
            fps = self.clock.get_fps()
            self.debug_panel.draw(
                [self.cursor_l, self.cursor_r],
                [self.wheel_a, self.wheel_b],
                [self.knob_filter_a, self.knob_filter_b,
                 self.knob_eq_hi_a, self.knob_eq_hi_b],
                [self.fader_vol_a, self.fader_vol_b, self.crossfader],
                fps)

            pygame.display.flip()

        self.cap.release()
        pygame.quit()


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand DJ — Cursor Mode")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--flip-camera", action="store_true",
                        help="Flip camera 180° (upside-down mount)")
    args = parser.parse_args()
    app = HandDJApp(camera_idx=args.camera, flip_camera=args.flip_camera)
    app.run()
