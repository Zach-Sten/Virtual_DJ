"""
Hand DJ Controller - Dual Display / Projector Mode
====================================================
Splits output into two windows:
  1. PROJECTOR WINDOW — fullscreen DJ layout on your second display
     (the pico projector pointing down at the desk via mirror)
  2. DEBUG WINDOW — camera feed with hand tracking overlay on your laptop

The projector shows only the clean DJ board (no camera, no debug info).
Your laptop shows the webcam feed, tracking data, and control states.

Architecture:
  Laptop runs everything:
    - Captures USB webcam (overhead camera on the arm)
    - Runs MediaPipe hand tracking
    - Renders DJ layout → sends to projector (second display)
    - Renders debug view → shows on laptop screen

Requirements:
  pip install mediapipe opencv-python pygame numpy

Usage:
  python hand_dj_projector.py

  Options:
    --camera 0          Camera index (default 0)
    --projector 1       Display index for projector output (default 1)
    --no-projector      Run without second display (projector window on same screen)
    --flip-camera       Flip camera feed (if image is upside down)
"""

import cv2
import mediapipe as mp
import pygame
import numpy as np
import math
import sys
import time
import argparse
from dataclasses import dataclass, field
from typing import Optional, Tuple

# ─── Configuration ───────────────────────────────────────────────────────────

# Projector output (what gets projected onto desk)
PROJ_WIDTH = 854
PROJ_HEIGHT = 480

# Debug window (laptop screen) — wider for two-panel layout
DEBUG_WIDTH = 1280
DEBUG_HEIGHT = 580

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS = 60

# Colors - Dark DJ aesthetic
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

# Top-down hand detection thresholds
# (tuned for overhead camera looking down at hands on a surface)
PINCH_THRESHOLD = 0.08        # Thumb-to-index distance for pinch (was 0.07)
FIST_THRESHOLD = 0.13         # Avg fingertip spread for fist (was 0.12)
FINGER_CURL_ANGLE = 140       # Degrees — above this = finger extended (was 160, too strict)
WHEEL_ZONE_PX = 130           # Pixel radius for jogwheel interaction zone
KNOB_ZONE_PX = 65             # Pixel radius for knob interaction zone
FADER_ZONE_PX = 45            # Pixel radius for fader interaction zone
LANDMARK_SMOOTH = 0.25        # EMA smoothing factor (lower = more responsive)


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class JogWheel:
    """Represents a DJ jogwheel."""
    angle: float = 0.0
    velocity: float = 0.0
    grabbed: bool = False
    last_hand_angle: float = 0.0
    deck_name: str = "A"
    color: Tuple[int, int, int] = (0, 220, 255)
    spin_history: list = field(default_factory=list)

    # Screen position (set by layout)
    cx: int = 0
    cy: int = 0
    radius: int = 100

    def update(self, dt: float):
        if not self.grabbed:
            self.velocity *= 0.97
            self.angle += self.velocity * dt
        self.angle = self.angle % 360

    def get_rpm(self) -> float:
        return abs(self.velocity) / 6.0


@dataclass
class Knob:
    """Represents a DJ knob/pot."""
    value: float = 0.5
    grabbed: bool = False
    last_rotation: float = 0.0
    label: str = "FILTER"
    color: Tuple[int, int, int] = (0, 220, 255)
    smoothed_value: float = 0.5

    # Screen position (set by layout)
    cx: int = 0
    cy: int = 0
    radius: int = 30

    def update(self):
        self.smoothed_value += (self.value - self.smoothed_value) * 0.15


@dataclass
class Fader:
    """Represents a DJ fader/slider."""
    value: float = 0.5          # 0.0 to 1.0
    grabbed: bool = False
    last_y: float = 0.0
    label: str = "VOLUME"
    color: Tuple[int, int, int] = (0, 220, 255)
    smoothed_value: float = 0.5
    vertical: bool = True

    # Screen position (set by layout)
    cx: int = 0
    cy: int = 0
    width: int = 20
    height: int = 100

    def update(self):
        self.smoothed_value += (self.value - self.smoothed_value) * 0.15


# ─── Hand Analyzer (Top-Down Optimized) ─────────────────────────────────────

class HandAnalyzer:
    """Analyzes hand landmarks with smoothing, angle-based gestures,
    and tuned for overhead (top-down) camera perspective."""

    # Finger joint chains: (MCP, PIP, DIP, TIP) landmark indices
    FINGER_JOINTS = {
        'index':  (5, 6, 7, 8),
        'middle': (9, 10, 11, 12),
        'ring':   (13, 14, 15, 16),
        'pinky':  (17, 18, 19, 20),
    }
    # Thumb is special: (CMC, MCP, IP, TIP)
    THUMB_JOINTS = (1, 2, 3, 4)

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,              # Lite model — better FPS on laptop
            min_detection_confidence=0.5,     # ← FIX 2: Lowered from 0.65
            min_tracking_confidence=0.4,      # ← FIX 2: Lowered from 0.55
        )
        self.mp_draw = mp.solutions.drawing_utils

        # ── FIX 3: Landmark smoothing ──
        # Store smoothed landmarks per hand (keyed by handedness)
        self._smooth_landmarks = {}  # "Left" / "Right" → list of (x,y,z)

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def _smooth(self, hand_key: str, landmarks) -> list:
        """Apply exponential moving average to landmark positions."""
        raw = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]

        if hand_key not in self._smooth_landmarks:
            self._smooth_landmarks[hand_key] = raw
            return raw

        prev = self._smooth_landmarks[hand_key]
        alpha = LANDMARK_SMOOTH
        smoothed = []
        for i in range(len(raw)):
            sx = prev[i][0] * alpha + raw[i][0] * (1 - alpha)
            sy = prev[i][1] * alpha + raw[i][1] * (1 - alpha)
            sz = prev[i][2] * alpha + raw[i][2] * (1 - alpha)
            smoothed.append((sx, sy, sz))

        self._smooth_landmarks[hand_key] = smoothed
        return smoothed

    def get_smoothed_landmarks(self, landmarks, hand_key: str = "unknown") -> list:
        """Get smoothed (x,y,z) tuples for all 21 landmarks."""
        return self._smooth(hand_key, landmarks)

    def get_hand_center(self, landmarks, smoothed=None) -> Tuple[float, float]:
        """Palm center from wrist + middle MCP."""
        if smoothed:
            wx, wy, _ = smoothed[0]
            mx, my, _ = smoothed[9]
        else:
            wx, wy = landmarks.landmark[0].x, landmarks.landmark[0].y
            mx, my = landmarks.landmark[9].x, landmarks.landmark[9].y
        return (wx + mx) / 2, (wy + my) / 2

    def get_index_tip(self, landmarks, smoothed=None) -> Tuple[float, float]:
        if smoothed:
            return smoothed[8][0], smoothed[8][1]
        return landmarks.landmark[8].x, landmarks.landmark[8].y

    def get_hand_rotation(self, landmarks, smoothed=None) -> float:
        """Rotation from wrist to middle fingertip."""
        if smoothed:
            wx, wy, _ = smoothed[0]
            mx, my, _ = smoothed[12]
        else:
            wx, wy = landmarks.landmark[0].x, landmarks.landmark[0].y
            mx, my = landmarks.landmark[12].x, landmarks.landmark[12].y
        return math.degrees(math.atan2(my - wy, mx - wx))

    # ── FIX 4: Angle-based finger curl detection ──

    def _angle_at_joint(self, smoothed, a_idx, b_idx, c_idx) -> float:
        """Calculate angle at point B formed by segments A→B and B→C.
        Returns degrees (0-180). Straight finger ≈ 170-180°, curled ≈ 30-90°."""
        ax, ay, _ = smoothed[a_idx]
        bx, by, _ = smoothed[b_idx]
        cx, cy, _ = smoothed[c_idx]

        ba = (ax - bx, ay - by)
        bc = (cx - bx, cy - by)

        dot = ba[0]*bc[0] + ba[1]*bc[1]
        mag_ba = math.sqrt(ba[0]**2 + ba[1]**2) + 1e-8
        mag_bc = math.sqrt(bc[0]**2 + bc[1]**2) + 1e-8

        cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
        return math.degrees(math.acos(cos_angle))

    def is_finger_extended(self, smoothed, finger_name: str) -> bool:
        """Check if a finger is extended using joint angles.
        More reliable than distance-based approach."""
        if finger_name == 'thumb':
            joints = self.THUMB_JOINTS
        else:
            joints = self.FINGER_JOINTS[finger_name]

        # Check angle at PIP joint (middle joint)
        angle_pip = self._angle_at_joint(smoothed, joints[0], joints[1], joints[2])
        # Check angle at DIP joint (last joint before tip)
        angle_dip = self._angle_at_joint(smoothed, joints[1], joints[2], joints[3])

        # Both joints should be relatively straight for "extended"
        return angle_pip > FINGER_CURL_ANGLE and angle_dip > FINGER_CURL_ANGLE

    def count_extended_fingers(self, smoothed) -> dict:
        """Returns dict of which fingers are extended."""
        result = {}
        for name in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            result[name] = self.is_finger_extended(smoothed, name)
        return result

    def is_fist(self, smoothed) -> bool:
        """All four fingers curled (thumb doesn't matter)."""
        fingers = self.count_extended_fingers(smoothed)
        curled = sum(1 for f in ['index','middle','ring','pinky'] if not fingers[f])
        return curled >= 3  # At least 3 of 4 fingers curled

    def is_open_palm(self, smoothed) -> bool:
        """At least 3 fingers extended, not pinching."""
        fingers = self.count_extended_fingers(smoothed)
        extended = sum(1 for f in ['index','middle','ring','pinky'] if fingers[f])
        return extended >= 3 and not self.is_pinch(smoothed)

    def is_pointing(self, smoothed) -> bool:
        """Index extended, other three curled."""
        fingers = self.count_extended_fingers(smoothed)
        index_up = fingers['index']
        others_down = sum(1 for f in ['middle','ring','pinky'] if not fingers[f])
        return index_up and others_down >= 2

    def is_pinch(self, smoothed) -> bool:
        """Thumb tip close to index tip."""
        tx, ty, _ = smoothed[4]
        ix, iy, _ = smoothed[8]
        dist = math.sqrt((tx - ix)**2 + (ty - iy)**2)
        return dist < PINCH_THRESHOLD

    def classify_handedness(self, label: str) -> str:
        return "Right" if label == "Left" else "Left"

    def draw_landmarks(self, frame, results):
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hl, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 200), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(0, 150, 255), thickness=1),
                )
        return frame


# ─── Projector Renderer (Clean DJ Layout) ───────────────────────────────────

class ProjectorRenderer:
    """Renders the clean DJ layout for the projector output.
    No camera feed, no debug info — just the board."""

    def __init__(self, surface: pygame.Surface):
        self.s = surface
        self.font_label = pygame.font.SysFont("Helvetica", 14, bold=True)
        self.font_small = pygame.font.SysFont("Helvetica", 11)
        self.font_tiny = pygame.font.SysFont("Helvetica", 9)

    def clear(self):
        self.s.fill(BG_COLOR)

    def draw_jogwheel(self, wheel: JogWheel):
        cx, cy, r = wheel.cx, wheel.cy, wheel.radius

        # Glow when grabbed
        if wheel.grabbed:
            for i in range(3):
                c = tuple(min(255, v // (i + 2)) for v in wheel.color)
                pygame.draw.circle(self.s, c, (cx, cy), r + 6 - i * 2, 2)

        # Platter
        pygame.draw.circle(self.s, JOGWHEEL_BG, (cx, cy), r)
        pygame.draw.circle(self.s, KNOB_RING, (cx, cy), r, 2)

        # Rotating lines
        for i in range(12):
            a = math.radians(wheel.angle + i * 30)
            x1 = cx + math.cos(a) * (r * 0.3)
            y1 = cy + math.sin(a) * (r * 0.3)
            x2 = cx + math.cos(a) * (r * 0.85)
            y2 = cy + math.sin(a) * (r * 0.85)
            color = wheel.color if i % 3 == 0 else ACCENT_DIM
            pygame.draw.line(self.s, color, (x1, y1), (x2, y2), 2 if i % 3 == 0 else 1)

        # Center
        pygame.draw.circle(self.s, wheel.color, (cx, cy), 6)
        pygame.draw.circle(self.s, BG_COLOR, (cx, cy), 3)

        # Position indicator
        ia = math.radians(wheel.angle)
        ix = cx + math.cos(ia) * (r * 0.92)
        iy = cy + math.sin(ia) * (r * 0.92)
        pygame.draw.circle(self.s, wheel.color, (int(ix), int(iy)), 4)

        # Label
        lbl = self.font_label.render(f"DECK {wheel.deck_name}", True, wheel.color)
        self.s.blit(lbl, (cx - lbl.get_width() // 2, cy - r - 20))

    def draw_knob(self, knob: Knob):
        cx, cy, r = knob.cx, knob.cy, knob.radius

        pygame.draw.circle(self.s, KNOB_BG, (cx, cy), r)

        # Background arc
        for a in range(0, 270, 2):
            angle = math.radians(135 + a)
            x = cx + math.cos(angle) * (r - 3)
            y = cy + math.sin(angle) * (r - 3)
            pygame.draw.circle(self.s, KNOB_RING, (int(x), int(y)), 2)

        # Value arc
        for a in range(0, int(270 * knob.smoothed_value), 2):
            angle = math.radians(135 + a)
            x = cx + math.cos(angle) * (r - 3)
            y = cy + math.sin(angle) * (r - 3)
            pygame.draw.circle(self.s, knob.color, (int(x), int(y)), 2)

        # Pointer
        pa = math.radians(135 + 270 * knob.smoothed_value)
        px = cx + math.cos(pa) * (r * 0.6)
        py = cy + math.sin(pa) * (r * 0.6)
        pygame.draw.line(self.s, ACCENT_WHITE, (cx, cy), (int(px), int(py)), 2)

        pygame.draw.circle(self.s, knob.color if knob.grabbed else ACCENT_DIM, (cx, cy), 4)

        lbl = self.font_tiny.render(knob.label, True, knob.color)
        self.s.blit(lbl, (cx - lbl.get_width() // 2, cy + r + 6))

    def draw_fader(self, fader: Fader):
        cx, cy = fader.cx, fader.cy
        w, h = fader.width, fader.height

        # Track
        track_rect = (cx - w // 2, cy - h // 2, w, h)
        pygame.draw.rect(self.s, KNOB_BG, track_rect, border_radius=4)
        pygame.draw.rect(self.s, KNOB_RING, track_rect, 1, border_radius=4)

        # Fill
        fill_h = int(h * fader.smoothed_value)
        fill_rect = (cx - w // 2 + 2, cy + h // 2 - fill_h, w - 4, fill_h)
        if fill_h > 0:
            pygame.draw.rect(self.s, fader.color, fill_rect, border_radius=2)

        # Thumb
        thumb_y = cy + h // 2 - int(h * fader.smoothed_value)
        pygame.draw.rect(self.s, ACCENT_WHITE if fader.grabbed else ACCENT_DIM,
                         (cx - w // 2 - 2, thumb_y - 4, w + 4, 8), border_radius=3)

        lbl = self.font_tiny.render(fader.label, True, fader.color)
        self.s.blit(lbl, (cx - lbl.get_width() // 2, cy + h // 2 + 8))

    def draw_crossfader(self, fader: Fader):
        cx, cy = fader.cx, fader.cy
        w, h = fader.width, fader.height  # width is the long axis here

        # Horizontal track
        track_rect = (cx - w // 2, cy - h // 2, w, h)
        pygame.draw.rect(self.s, KNOB_BG, track_rect, border_radius=4)
        pygame.draw.rect(self.s, KNOB_RING, track_rect, 1, border_radius=4)

        # Thumb position
        thumb_x = cx - w // 2 + int(w * fader.smoothed_value)
        pygame.draw.rect(self.s, ACCENT_WHITE if fader.grabbed else ACCENT_DIM,
                         (thumb_x - 6, cy - h // 2 - 2, 12, h + 4), border_radius=3)

        # A/B labels
        a_lbl = self.font_tiny.render("A", True, ACCENT_CYAN)
        b_lbl = self.font_tiny.render("B", True, ACCENT_MAGENTA)
        self.s.blit(a_lbl, (cx - w // 2 - 14, cy - 5))
        self.s.blit(b_lbl, (cx + w // 2 + 6, cy - 5))

        lbl = self.font_tiny.render(fader.label, True, TEXT_DIM)
        self.s.blit(lbl, (cx - lbl.get_width() // 2, cy + h // 2 + 6))

    def draw_hand_skeletons(self, results):
        """Draw MediaPipe hand skeletons mapped to the projector surface."""
        if not results or not results.multi_hand_landmarks:
            return

        # MediaPipe hand connections
        connections = [
            (0,1),(1,2),(2,3),(3,4),       # Thumb
            (0,5),(5,6),(6,7),(7,8),       # Index
            (5,9),(9,10),(10,11),(11,12),   # Middle
            (9,13),(13,14),(14,15),(15,16), # Ring
            (13,17),(17,18),(18,19),(19,20),# Pinky
            (0,17),                         # Palm base
        ]

        for hand_idx, hand_lm in enumerate(results.multi_hand_landmarks):
            # Pick color based on which side of screen
            cx_norm = (hand_lm.landmark[0].x + hand_lm.landmark[9].x) / 2
            if cx_norm <= 0.5:
                bone_color = (0, 255, 220)   # Cyan-green for left/Deck A
                joint_color = (0, 180, 255)
            else:
                bone_color = (255, 0, 200)   # Magenta-pink for right/Deck B
                joint_color = (255, 100, 255)

            # Draw bones (connections)
            for c1, c2 in connections:
                lm1 = hand_lm.landmark[c1]
                lm2 = hand_lm.landmark[c2]
                x1 = int(lm1.x * PROJ_WIDTH)
                y1 = int(lm1.y * PROJ_HEIGHT)
                x2 = int(lm2.x * PROJ_WIDTH)
                y2 = int(lm2.y * PROJ_HEIGHT)
                pygame.draw.line(self.s, bone_color, (x1, y1), (x2, y2), 2)

            # Draw joints
            for i, lm in enumerate(hand_lm.landmark):
                x = int(lm.x * PROJ_WIDTH)
                y = int(lm.y * PROJ_HEIGHT)
                # Fingertips are bigger
                if i in (4, 8, 12, 16, 20):
                    pygame.draw.circle(self.s, (255, 255, 255), (x, y), 5)
                    pygame.draw.circle(self.s, joint_color, (x, y), 3)
                else:
                    pygame.draw.circle(self.s, joint_color, (x, y), 3)

            # Draw palm center indicator
            pcx = int(cx_norm * PROJ_WIDTH)
            pcy_norm = (hand_lm.landmark[0].y + hand_lm.landmark[9].y) / 2
            pcy = int(pcy_norm * PROJ_HEIGHT)
            pygame.draw.circle(self.s, (255, 255, 100), (pcx, pcy), 8, 2)

    def draw_interaction_zones(self, wheels, knobs, faders):
        """Draw faint dashed circles/rects showing where grab zones are.
        This helps the user see exactly where they need to position their hand."""
        zone_color = (40, 50, 40)       # Very faint green
        zone_active = (60, 100, 60)     # Slightly brighter when grabbed

        # Jogwheel zones
        for w in wheels:
            c = zone_active if w.grabbed else zone_color
            # Draw dashed circle for interaction zone
            for a in range(0, 360, 8):
                angle = math.radians(a)
                x = w.cx + math.cos(angle) * WHEEL_ZONE_PX
                y = w.cy + math.sin(angle) * WHEEL_ZONE_PX
                pygame.draw.circle(self.s, c, (int(x), int(y)), 1)

        # Knob zones
        for k in knobs:
            c = zone_active if k.grabbed else zone_color
            for a in range(0, 360, 10):
                angle = math.radians(a)
                x = k.cx + math.cos(angle) * KNOB_ZONE_PX
                y = k.cy + math.sin(angle) * KNOB_ZONE_PX
                pygame.draw.circle(self.s, c, (int(x), int(y)), 1)

        # Fader zones
        for f in faders:
            c = zone_active if f.grabbed else zone_color
            if f.vertical:
                rect = (f.cx - FADER_ZONE_PX, f.cy - f.height // 2 - 30,
                        FADER_ZONE_PX * 2, f.height + 60)
            else:
                rect = (f.cx - f.width // 2 - 20, f.cy - FADER_ZONE_PX,
                        f.width + 40, FADER_ZONE_PX * 2)
            pygame.draw.rect(self.s, c, rect, 1, border_radius=4)


# ─── Debug Renderer (Laptop Screen) ─────────────────────────────────────────

class DebugRenderer:
    """Renders the debug view on the laptop screen.
    Left panel: DJ board preview with hand skeletons overlaid.
    Right panel: green-on-black debug log with control states."""

    PANEL_DIVIDER_X = 860   # Where the right panel starts
    LOG_BG = (10, 14, 10)   # Dark green-black background
    LOG_GREEN = (0, 255, 80)
    LOG_DIM = (0, 140, 50)
    LOG_BRIGHT = (80, 255, 140)
    LOG_CYAN = (0, 220, 200)
    LOG_MAGENTA = (255, 60, 180)
    LOG_WHITE = (200, 255, 200)
    LOG_GRABBED = (255, 255, 80)

    def __init__(self, surface: pygame.Surface):
        self.s = surface
        self.font_title = pygame.font.SysFont("Helvetica", 18, bold=True)
        self.font_log = pygame.font.SysFont("Courier", 13)
        self.font_log_bold = pygame.font.SysFont("Courier", 13, bold=True)
        self.font_log_small = pygame.font.SysFont("Courier", 11)
        self.font_header = pygame.font.SysFont("Courier", 10, bold=True)
        self.log_lines = []  # Rolling log of events
        self.max_log_lines = 18

    def clear(self):
        self.s.fill((16, 16, 24))

    def draw_dj_preview(self, proj_surface: pygame.Surface):
        """Draw a scaled copy of the projector surface on the left panel."""
        # Scale projector surface to fit left panel
        target_w = self.PANEL_DIVIDER_X - 20
        target_h = DEBUG_HEIGHT - 60
        # Maintain aspect ratio
        scale = min(target_w / PROJ_WIDTH, target_h / PROJ_HEIGHT)
        sw = int(PROJ_WIDTH * scale)
        sh = int(PROJ_HEIGHT * scale)
        x = 10
        y = 50

        scaled = pygame.transform.smoothscale(proj_surface, (sw, sh))
        # Border
        pygame.draw.rect(self.s, (40, 40, 60), (x - 2, y - 2, sw + 4, sh + 4), 2)
        self.s.blit(scaled, (x, y))

        # Label
        lbl = self.font_header.render("DJ BOARD + HAND TRACKING", True, ACCENT_CYAN)
        self.s.blit(lbl, (x, y - 16))

    def add_log_event(self, msg: str):
        """Add a timestamped event to the rolling log."""
        ts = time.strftime("%H:%M:%S")
        self.log_lines.append(f"[{ts}] {msg}")
        if len(self.log_lines) > self.max_log_lines:
            self.log_lines.pop(0)

    def draw_debug_panel(self, left_det, right_det, left_gest, right_gest,
                          wheels, knobs, faders, fps):
        """Draw the right-side green debug log panel."""
        px = self.PANEL_DIVIDER_X
        pw = DEBUG_WIDTH - px
        py = 0

        # Panel background
        pygame.draw.rect(self.s, self.LOG_BG, (px, py, pw, DEBUG_HEIGHT))
        # Divider line
        pygame.draw.line(self.s, (0, 80, 40), (px, 0), (px, DEBUG_HEIGHT), 2)

        # Header
        x = px + 12
        y = 10

        title = self.font_title.render("DEBUG LOG", True, self.LOG_GREEN)
        self.s.blit(title, (x, y))
        fps_txt = self.font_log_small.render(f"{fps:.0f} FPS", True, self.LOG_DIM)
        self.s.blit(fps_txt, (px + pw - 60, y + 4))
        y += 30

        # Separator
        pygame.draw.line(self.s, (0, 60, 30), (px + 8, y), (px + pw - 8, y), 1)
        y += 8

        # Hand status
        sec_title = self.font_header.render("── HANDS ──", True, self.LOG_DIM)
        self.s.blit(sec_title, (x, y))
        y += 18

        l_color = self.LOG_BRIGHT if left_det else self.LOG_DIM
        r_color = self.LOG_BRIGHT if right_det else self.LOG_DIM
        l_status = f"L: {left_gest:>6}" if left_det else "L:    ---"
        r_status = f"R: {right_gest:>6}" if right_det else "R:    ---"

        dot_l = "●" if left_det else "○"
        dot_r = "●" if right_det else "○"
        self.s.blit(self.font_log.render(f" {dot_l} {l_status}", True, l_color), (x, y))
        y += 17
        self.s.blit(self.font_log.render(f" {dot_r} {r_status}", True, r_color), (x, y))
        y += 24

        # Decks
        sec_title = self.font_header.render("── DECKS ──", True, self.LOG_DIM)
        self.s.blit(sec_title, (x, y))
        y += 18

        for w in wheels:
            grabbed = w.grabbed
            color = self.LOG_GRABBED if grabbed else self.LOG_GREEN
            tag = " ★" if grabbed else ""
            line = f" DECK {w.deck_name}: {w.angle:6.1f}° {w.velocity:+6.1f}{tag}"
            self.s.blit(self.font_log.render(line, True, color), (x, y))
            y += 17
        y += 8

        # Knobs
        sec_title = self.font_header.render("── KNOBS ──", True, self.LOG_DIM)
        self.s.blit(sec_title, (x, y))
        y += 18

        for k in knobs:
            grabbed = k.grabbed
            color = self.LOG_GRABBED if grabbed else self.LOG_GREEN
            tag = " ★" if grabbed else ""
            bar_len = 10
            filled = int(bar_len * k.smoothed_value)
            bar = "█" * filled + "░" * (bar_len - filled)
            line = f" {k.label:>10}: {bar} {k.smoothed_value*100:4.0f}%{tag}"
            self.s.blit(self.font_log.render(line, True, color), (x, y))
            y += 17
        y += 8

        # Faders
        sec_title = self.font_header.render("── FADERS ──", True, self.LOG_DIM)
        self.s.blit(sec_title, (x, y))
        y += 18

        for f in faders:
            grabbed = f.grabbed
            color = self.LOG_GRABBED if grabbed else self.LOG_GREEN
            tag = " ★" if grabbed else ""
            bar_len = 10
            filled = int(bar_len * f.smoothed_value)
            bar = "█" * filled + "░" * (bar_len - filled)
            line = f" {f.label:>10}: {bar} {f.smoothed_value*100:4.0f}%{tag}"
            self.s.blit(self.font_log.render(line, True, color), (x, y))
            y += 17
        y += 12

        # Event log
        pygame.draw.line(self.s, (0, 60, 30), (px + 8, y), (px + pw - 8, y), 1)
        y += 6
        sec_title = self.font_header.render("── EVENT LOG ──", True, self.LOG_DIM)
        self.s.blit(sec_title, (x, y))
        y += 16

        for line in self.log_lines[-8:]:
            self.s.blit(self.font_log_small.render(line, True, self.LOG_DIM), (x, y))
            y += 14

    def draw_header(self):
        title = self.font_title.render("HAND DJ — DEBUG VIEW", True, ACCENT_CYAN)
        self.s.blit(title, (10, 10))

        instructions = self.font_log_small.render(
            "Q/ESC=Quit  |  F=Toggle Fullscreen Projector  |  C=Calibrate", True, (80, 80, 110))
        self.s.blit(instructions, (10, 32))


# ─── Main Application ───────────────────────────────────────────────────────

class HandDJApp:
    """Dual-display DJ controller with projector output + debug view."""

    def __init__(self, camera_idx=0, projector_display=1,
                 no_projector=False, flip_camera=False):
        pygame.init()

        self.flip_camera = flip_camera
        self.no_projector = no_projector

        # ── Create windows ──
        # Debug window (laptop) — always on primary display
        self.debug_screen = pygame.display.set_mode(
            (DEBUG_WIDTH, DEBUG_HEIGHT))
        pygame.display.set_caption("Hand DJ — Debug View")

        # Projector surface (rendered offscreen, blitted or sent to second display)
        # For now we use an offscreen surface; in fullscreen projector mode
        # you'd create a second window on the projector display
        self.proj_surface = pygame.Surface((PROJ_WIDTH, PROJ_HEIGHT))

        # No second window — debug view shows DJ board preview on left panel
        self.use_cv_projector = False

        # ── Components ──
        self.hand_analyzer = HandAnalyzer()
        self.proj_renderer = ProjectorRenderer(self.proj_surface)
        self.debug_renderer = DebugRenderer(self.debug_screen)

        # ── DJ Controls ──
        self.wheel_a = JogWheel(deck_name="A", color=ACCENT_CYAN)
        self.wheel_b = JogWheel(deck_name="B", color=ACCENT_MAGENTA)

        self.knob_filter_a = Knob(label="FILTER A", color=ACCENT_CYAN)
        self.knob_filter_b = Knob(label="FILTER B", color=ACCENT_MAGENTA)
        self.knob_eq_hi_a = Knob(label="HI A", color=ACCENT_CYAN)
        self.knob_eq_hi_b = Knob(label="HI B", color=ACCENT_MAGENTA)

        self.fader_vol_a = Fader(label="VOL A", color=ACCENT_CYAN)
        self.fader_vol_b = Fader(label="VOL B", color=ACCENT_MAGENTA)
        self.crossfader = Fader(label="CROSSFADER", color=ACCENT_WHITE, vertical=False)

        self._compute_layout()

        # ── Camera ──
        self.cap = cv2.VideoCapture(camera_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not self.cap.isOpened():
            print(f"ERROR: Cannot open camera index {camera_idx}")
            sys.exit(1)

        self.clock = pygame.time.Clock()
        self.last_frame = None
        self.last_results = None
        self.running = True
        self.actual_fps = 60.0

        # Grab hysteresis counters — prevents flickering
        self._grab_frames = {}   # control_id → frames in grab state
        self._drop_frames = {}   # control_id → frames in drop state
        self.GRAB_DELAY = 1      # Grab on first frame (instant)
        self.DROP_DELAY = 3      # Hold grab for 3 frames after losing gesture

    def _compute_layout(self):
        """Position all controls on the projector surface."""
        pw, ph = PROJ_WIDTH, PROJ_HEIGHT

        # Jogwheels — large, centered vertically, left/right sides
        wr = 90
        self.wheel_a.cx, self.wheel_a.cy, self.wheel_a.radius = 140, ph // 2 - 20, wr
        self.wheel_b.cx, self.wheel_b.cy, self.wheel_b.radius = pw - 140, ph // 2 - 20, wr

        # Knobs — above jogwheels
        kr = 22
        self.knob_filter_a.cx, self.knob_filter_a.cy, self.knob_filter_a.radius = 140, 55, kr
        self.knob_filter_b.cx, self.knob_filter_b.cy, self.knob_filter_b.radius = pw - 140, 55, kr
        self.knob_eq_hi_a.cx, self.knob_eq_hi_a.cy, self.knob_eq_hi_a.radius = 60, 55, kr
        self.knob_eq_hi_b.cx, self.knob_eq_hi_b.cy, self.knob_eq_hi_b.radius = pw - 60, 55, kr

        # Volume faders — between jogwheels and center
        self.fader_vol_a.cx, self.fader_vol_a.cy = 300, ph // 2
        self.fader_vol_a.width, self.fader_vol_a.height = 18, 120
        self.fader_vol_b.cx, self.fader_vol_b.cy = pw - 300, ph // 2
        self.fader_vol_b.width, self.fader_vol_b.height = 18, 120

        # Crossfader — bottom center, horizontal
        self.crossfader.cx, self.crossfader.cy = pw // 2, ph - 40
        self.crossfader.width, self.crossfader.height = 200, 16

    def _all_wheels(self):
        return [self.wheel_a, self.wheel_b]

    def _all_knobs(self):
        return [self.knob_filter_a, self.knob_filter_b,
                self.knob_eq_hi_a, self.knob_eq_hi_b]

    def _all_faders(self):
        return [self.fader_vol_a, self.fader_vol_b, self.crossfader]

    def _should_grab(self, control_id: str, wants_grab: bool) -> bool:
        """Hysteresis: require gesture to hold for N frames before grab/drop.
        Prevents the rapid GRAB/DROP flickering."""
        if wants_grab:
            self._grab_frames[control_id] = self._grab_frames.get(control_id, 0) + 1
            self._drop_frames[control_id] = 0
            return self._grab_frames[control_id] >= self.GRAB_DELAY
        else:
            self._drop_frames[control_id] = self._drop_frames.get(control_id, 0) + 1
            self._grab_frames[control_id] = 0
            return self._drop_frames[control_id] < self.DROP_DELAY  # Stay grabbed until drop delay met

    def _process_hands(self, results):
        """Process hand landmarks and map to DJ controls.
        Uses smoothed landmarks and angle-based gesture detection."""
        left_det, right_det = False, False
        left_gest, right_gest = "---", "---"

        if not results.multi_hand_landmarks:
            for w in self._all_wheels(): w.grabbed = False
            for k in self._all_knobs(): k.grabbed = False
            for f in self._all_faders(): f.grabbed = False
            return left_det, right_det, left_gest, right_gest

        for idx, hand_lm in enumerate(results.multi_hand_landmarks):
            # Determine handedness for smoothing key
            hand_key = "unknown"
            if results.multi_handedness and idx < len(results.multi_handedness):
                hand_key = results.multi_handedness[idx].classification[0].label

            # Get smoothed landmarks
            smoothed = self.hand_analyzer.get_smoothed_landmarks(hand_lm, hand_key)

            # Hand center in normalized coords (smoothed)
            cx, cy = self.hand_analyzer.get_hand_center(hand_lm, smoothed)
            rotation = self.hand_analyzer.get_hand_rotation(hand_lm, smoothed)

            # Angle-based gesture detection (smoothed)
            pinch = self.hand_analyzer.is_pinch(smoothed)
            pointing = self.hand_analyzer.is_pointing(smoothed)
            open_palm = self.hand_analyzer.is_open_palm(smoothed)
            fist = self.hand_analyzer.is_fist(smoothed)

            # Map to projector pixel coords
            hpx = cx * PROJ_WIDTH
            hpy = cy * PROJ_HEIGHT

            # Determine gesture label
            if pinch:
                gesture = "PINCH"
            elif pointing:
                gesture = "POINT"
            elif open_palm:
                gesture = "PALM"
            elif fist:
                gesture = "FIST"
            else:
                gesture = "---"

            # Determine which side (left/right of projector surface)
            if cx <= 0.5:
                left_det = True
                left_gest = gesture
                wheel = self.wheel_a
                knobs = [self.knob_filter_a, self.knob_eq_hi_a]
                fader = self.fader_vol_a
            else:
                right_det = True
                right_gest = gesture
                wheel = self.wheel_b
                knobs = [self.knob_filter_b, self.knob_eq_hi_b]
                fader = self.fader_vol_b

            # Check interaction zones (with larger radii)
            wheel_dist = math.sqrt((hpx - wheel.cx)**2 + (hpy - wheel.cy)**2)
            in_wheel = wheel_dist < WHEEL_ZONE_PX

            closest_knob = None
            closest_knob_dist = float('inf')
            for k in knobs:
                d = math.sqrt((hpx - k.cx)**2 + (hpy - k.cy)**2)
                if d < KNOB_ZONE_PX and d < closest_knob_dist:
                    closest_knob = k
                    closest_knob_dist = d

            fader_dist_x = abs(hpx - fader.cx)
            fader_dist_y = abs(hpy - fader.cy)
            in_fader = fader_dist_x < FADER_ZONE_PX and fader_dist_y < fader.height // 2 + 30

            # Apply gestures to controls with hysteresis
            wants_wheel = open_palm and in_wheel
            wants_knob = pinch and closest_knob is not None
            wants_fader = pointing and in_fader

            wheel_id = f"wheel_{wheel.deck_name}"
            knob_id = f"knob_{closest_knob.label}" if closest_knob else "knob_none"
            fader_id = f"fader_{fader.label}"

            if wants_wheel and self._should_grab(wheel_id, True):
                # Jogwheel: track hand angle around wheel center
                hand_angle = math.degrees(math.atan2(hpy - wheel.cy, hpx - wheel.cx))
                if not wheel.grabbed:
                    wheel.grabbed = True
                    wheel.last_hand_angle = hand_angle
                else:
                    delta = hand_angle - wheel.last_hand_angle
                    if delta > 180: delta -= 360
                    elif delta < -180: delta += 360
                    wheel.angle += delta
                    wheel.velocity = delta * FPS
                    wheel.last_hand_angle = hand_angle
                for k in knobs: k.grabbed = False
                fader.grabbed = False

            elif wants_knob and self._should_grab(knob_id, True):
                k = closest_knob
                if not k.grabbed:
                    k.grabbed = True
                    k.last_rotation = rotation
                else:
                    delta_rot = rotation - k.last_rotation
                    if delta_rot > 180: delta_rot -= 360
                    elif delta_rot < -180: delta_rot += 360
                    k.value = max(0.0, min(1.0, k.value + delta_rot / 270.0))
                    k.last_rotation = rotation
                wheel.grabbed = False
                fader.grabbed = False
                for other_k in knobs:
                    if other_k is not k:
                        other_k.grabbed = False

            elif wants_fader and self._should_grab(fader_id, True):
                ix, iy = self.hand_analyzer.get_index_tip(hand_lm, smoothed)
                if not fader.grabbed:
                    fader.grabbed = True
                    fader.last_y = iy
                else:
                    delta_y = fader.last_y - iy
                    fader.value = max(0.0, min(1.0, fader.value + delta_y * 2.0))
                    fader.last_y = iy
                wheel.grabbed = False
                for k in knobs: k.grabbed = False

            else:
                # Only release if drop delay has been met
                if not self._should_grab(wheel_id, False):
                    wheel.grabbed = False
                if closest_knob and not self._should_grab(knob_id, False):
                    for k in knobs: k.grabbed = False
                if not self._should_grab(fader_id, False):
                    fader.grabbed = False

            # Crossfader: either hand, pinch near bottom center
            cf = self.crossfader
            cf_dist = math.sqrt((hpx - cf.cx)**2 + (hpy - cf.cy)**2)
            if pinch and cf_dist < 140:
                ix, iy = self.hand_analyzer.get_index_tip(hand_lm, smoothed)
                if not cf.grabbed:
                    cf.grabbed = True
                    cf.last_y = ix
                else:
                    delta_x = ix - cf.last_y
                    cf.value = max(0.0, min(1.0, cf.value + delta_x * 2.0))
                    cf.last_y = ix
            elif cf.grabbed and not pinch:
                cf.grabbed = False

        return left_det, right_det, left_gest, right_gest

    def run(self):
        print("\n" + "=" * 56)
        print("  HAND DJ CONTROLLER — DEBUG VIEW")
        print("  DJ board + hand tracking on left")
        print("  Debug log on right")
        print("  Press Q or ESC to quit")
        print("=" * 56 + "\n")

        while self.running:
            dt = self.clock.tick(FPS) / 1000.0

            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        self.running = False

            # Camera
            ret, frame = self.cap.read()
            if ret:
                if self.flip_camera:
                    frame = cv2.flip(frame, -1)  # 180° rotation for upside-down mount
                else:
                    frame = cv2.flip(frame, 1)   # Mirror for front-facing

                results = self.hand_analyzer.process(frame)
                self.last_results = results
                old_grabs = {
                    'wheel_a': self.wheel_a.grabbed, 'wheel_b': self.wheel_b.grabbed,
                    **{k.label: k.grabbed for k in self._all_knobs()},
                    **{f.label: f.grabbed for f in self._all_faders()},
                }
                left_det, right_det, left_gest, right_gest = self._process_hands(results)
                # Log grab/release events
                new_grabs = {
                    'wheel_a': self.wheel_a.grabbed, 'wheel_b': self.wheel_b.grabbed,
                    **{k.label: k.grabbed for k in self._all_knobs()},
                    **{f.label: f.grabbed for f in self._all_faders()},
                }
                for key in old_grabs:
                    if not old_grabs[key] and new_grabs[key]:
                        self.debug_renderer.add_log_event(f"GRAB  {key}")
                    elif old_grabs[key] and not new_grabs[key]:
                        self.debug_renderer.add_log_event(f"DROP  {key}")

                frame = self.hand_analyzer.draw_landmarks(frame, results)
                self.last_frame = frame
            else:
                left_det, right_det = False, False
                left_gest, right_gest = "---", "---"

            # Update controls
            for w in self._all_wheels(): w.update(dt)
            for k in self._all_knobs(): k.update()
            for f in self._all_faders(): f.update()

            # ── Render projector surface ──
            self.proj_renderer.clear()
            # Draw interaction zones FIRST (underneath everything)
            self.proj_renderer.draw_interaction_zones(
                self._all_wheels(), self._all_knobs(), self._all_faders())
            self.proj_renderer.draw_jogwheel(self.wheel_a)
            self.proj_renderer.draw_jogwheel(self.wheel_b)
            for k in self._all_knobs():
                self.proj_renderer.draw_knob(k)
            self.proj_renderer.draw_fader(self.fader_vol_a)
            self.proj_renderer.draw_fader(self.fader_vol_b)
            self.proj_renderer.draw_crossfader(self.crossfader)
            # Draw hand skeletons on top of the DJ board
            self.proj_renderer.draw_hand_skeletons(self.last_results)

            # ── Render debug view (single window, two-panel layout) ──
            self.actual_fps = self.clock.get_fps()
            self.debug_renderer.clear()
            self.debug_renderer.draw_header()
            # Left panel: scaled DJ board with skeletons already drawn
            self.debug_renderer.draw_dj_preview(self.proj_surface)
            # Right panel: green debug log
            self.debug_renderer.draw_debug_panel(
                left_det, right_det, left_gest, right_gest,
                self._all_wheels(), self._all_knobs(), self._all_faders(),
                self.actual_fps)

            pygame.display.flip()

        # Cleanup
        self.cap.release()
        pygame.quit()


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand DJ Controller — Projector Mode")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--projector", type=int, default=1, help="Display index for projector")
    parser.add_argument("--no-projector", action="store_true", help="Run without second display")
    parser.add_argument("--flip-camera", action="store_true",
                        help="Flip camera 180° (for upside-down mounted cameras)")
    args = parser.parse_args()

    app = HandDJApp(
        camera_idx=args.camera,
        projector_display=args.projector,
        no_projector=args.no_projector,
        flip_camera=args.flip_camera,
    )
    app.run()
