"""
Hand DJ Controller - Air DJ Prototype
======================================
Uses MediaPipe Hands to track hand gestures and map them to
a virtual DJ interface with 2 jogwheels and 2 knobs.

Controls:
  - Left hand controls Deck A (left jogwheel + left knob)
  - Right hand controls Deck B (right jogwheel + right knob)
  - Closed fist near jogwheel area = grab jogwheel, rotate hand to spin
  - Index finger near knob area = grab knob, move up/down to turn

Requirements:
  pip install mediapipe opencv-python pygame numpy

Usage:
  python hand_dj_controller.py
"""

import cv2
import mediapipe as mp
import pygame
import numpy as np
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

# ─── Configuration ───────────────────────────────────────────────────────────

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS = 60

# Colors - Dark DJ aesthetic
BG_COLOR = (12, 12, 18)
DECK_BG = (22, 22, 32)
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
WEBCAM_BORDER = (40, 40, 60)

# Hand detection thresholds
PINCH_THRESHOLD = 0.07       # Distance for pinch detection
FIST_THRESHOLD = 0.12        # Average finger distance for fist detection
WHEEL_ZONE_PX = 160          # Pixel radius around jogwheel center to activate
KNOB_ZONE_PX  = 90           # Pixel radius around knob center to activate


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

    def update(self, dt: float):
        if not self.grabbed:
            # Apply friction
            self.velocity *= 0.97
            self.angle += self.velocity * dt
        # Normalize angle
        self.angle = self.angle % 360

    def get_rpm(self) -> float:
        return abs(self.velocity) / 6.0  # Rough RPM estimate


@dataclass
class Knob:
    """Represents a DJ knob/pot."""
    value: float = 0.5          # 0.0 to 1.0
    grabbed: bool = False
    last_rotation: float = 0.0
    label: str = "FILTER"
    color: Tuple[int, int, int] = (0, 220, 255)
    smoothed_value: float = 0.5

    def update(self):
        # Smooth the value
        self.smoothed_value += (self.value - self.smoothed_value) * 0.15


# ─── Hand Analyzer ───────────────────────────────────────────────────────────

class HandAnalyzer:
    """Analyzes MediaPipe hand landmarks for DJ gestures."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame):
        """Process a frame and return hand data."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        return results

    def get_hand_center(self, landmarks) -> Tuple[float, float]:
        """Get the center of the palm (wrist + middle finger MCP average)."""
        wrist = landmarks.landmark[0]
        middle_mcp = landmarks.landmark[9]
        cx = (wrist.x + middle_mcp.x) / 2
        cy = (wrist.y + middle_mcp.y) / 2
        return cx, cy

    def get_index_tip(self, landmarks) -> Tuple[float, float]:
        """Get index fingertip position."""
        tip = landmarks.landmark[8]
        return tip.x, tip.y

    def get_hand_rotation(self, landmarks) -> float:
        """Get hand rotation angle based on wrist to middle finger direction."""
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        dx = middle_tip.x - wrist.x
        dy = middle_tip.y - wrist.y
        return math.degrees(math.atan2(dy, dx))

    def is_fist(self, landmarks) -> bool:
        """Detect if hand is making a fist (fingers curled)."""
        # Check distance from each fingertip to palm center
        palm_x, palm_y = self.get_hand_center(landmarks)
        tips = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
        total_dist = 0
        for tip_id in tips:
            tip = landmarks.landmark[tip_id]
            dist = math.sqrt((tip.x - palm_x)**2 + (tip.y - palm_y)**2)
            total_dist += dist
        avg_dist = total_dist / len(tips)
        return avg_dist < FIST_THRESHOLD

    def is_pointing(self, landmarks) -> bool:
        """Index finger extended, other three curled."""
        palm_x, palm_y = self.get_hand_center(landmarks)
        index_tip = landmarks.landmark[8]
        index_dist = math.sqrt((index_tip.x - palm_x)**2 + (index_tip.y - palm_y)**2)
        other_tips = [12, 16, 20]
        other_avg = sum(
            math.sqrt((landmarks.landmark[t].x - palm_x)**2 + (landmarks.landmark[t].y - palm_y)**2)
            for t in other_tips
        ) / len(other_tips)
        return index_dist > 0.15 and other_avg < FIST_THRESHOLD

    def is_open_palm(self, landmarks) -> bool:
        """All fingers extended (not a fist, not pointing, not pinching)."""
        palm_x, palm_y = self.get_hand_center(landmarks)
        tips = [8, 12, 16, 20]
        total_dist = 0
        for tip_id in tips:
            tip = landmarks.landmark[tip_id]
            dist = math.sqrt((tip.x - palm_x)**2 + (tip.y - palm_y)**2)
            total_dist += dist
        avg_dist = total_dist / len(tips)
        return avg_dist >= FIST_THRESHOLD

    def is_pinch(self, landmarks) -> bool:
        """Detect pinch: thumb tip close to index tip."""
        thumb = landmarks.landmark[4]
        index = landmarks.landmark[8]
        dist = math.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2)
        return dist < PINCH_THRESHOLD

    def draw_landmarks_on_frame(self, frame, results):
        """Draw hand landmarks on the camera frame."""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 200), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(0, 150, 255), thickness=1),
                )
        return frame


# ─── DJ Interface Renderer ──────────────────────────────────────────────────

class DJRenderer:
    """Renders the DJ controller interface using Pygame."""

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        pygame.font.init()
        self.font_large = pygame.font.SysFont("Helvetica", 28, bold=True)
        self.font_medium = pygame.font.SysFont("Helvetica", 18)
        self.font_small = pygame.font.SysFont("Helvetica", 13)
        self.font_tiny = pygame.font.SysFont("Helvetica", 11)
        self.frame_count = 0

    def draw_background(self, frame=None):
        """Draw background - webcam feed full-screen or solid color fallback."""
        if frame is not None:
            frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(np.rot90(frame_rgb, 1))
            surf = pygame.transform.flip(surf, True, False)
            self.screen.blit(surf, (0, 0))
            # Dark overlay for readability
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            overlay.fill((12, 12, 18, 150))
            self.screen.blit(overlay, (0, 0))
        else:
            self.screen.fill(BG_COLOR)
            for x in range(0, WINDOW_WIDTH, 40):
                pygame.draw.line(self.screen, (16, 16, 24), (x, 0), (x, WINDOW_HEIGHT), 1)
            for y in range(0, WINDOW_HEIGHT, 40):
                pygame.draw.line(self.screen, (16, 16, 24), (0, y), (WINDOW_WIDTH, y), 1)

    def draw_jogwheel(self, cx: int, cy: int, radius: int, wheel: JogWheel):
        """Draw a jogwheel at the given position."""
        # Outer ring glow
        glow_color = (*wheel.color[:3],)
        if wheel.grabbed:
            for i in range(3):
                alpha_color = tuple(min(255, c // (i + 2)) for c in glow_color)
                pygame.draw.circle(self.screen, alpha_color, (cx, cy), radius + 8 - i * 2, 2)

        # Main circle background
        pygame.draw.circle(self.screen, JOGWHEEL_BG, (cx, cy), radius)
        pygame.draw.circle(self.screen, KNOB_RING, (cx, cy), radius, 2)

        # Platter lines (rotating)
        num_lines = 12
        for i in range(num_lines):
            angle_rad = math.radians(wheel.angle + i * (360 / num_lines))
            inner_r = radius * 0.3
            outer_r = radius * 0.85
            x1 = cx + math.cos(angle_rad) * inner_r
            y1 = cy + math.sin(angle_rad) * inner_r
            x2 = cx + math.cos(angle_rad) * outer_r
            y2 = cy + math.sin(angle_rad) * outer_r
            line_color = ACCENT_DIM if i % 3 != 0 else wheel.color
            pygame.draw.line(self.screen, line_color, (x1, y1), (x2, y2), 2 if i % 3 == 0 else 1)

        # Center dot
        pygame.draw.circle(self.screen, wheel.color, (cx, cy), 8)
        pygame.draw.circle(self.screen, BG_COLOR, (cx, cy), 4)

        # Indicator dot (shows current angle)
        ind_angle = math.radians(wheel.angle)
        ind_x = cx + math.cos(ind_angle) * (radius * 0.92)
        ind_y = cy + math.sin(ind_angle) * (radius * 0.92)
        pygame.draw.circle(self.screen, wheel.color, (int(ind_x), int(ind_y)), 5)

        # RPM display
        rpm = wheel.get_rpm()
        rpm_text = self.font_small.render(f"{rpm:.1f} RPM", True, TEXT_DIM)
        self.screen.blit(rpm_text, (cx - rpm_text.get_width() // 2, cy + radius + 15))

        # Deck label
        label = self.font_medium.render(f"DECK {wheel.deck_name}", True, wheel.color)
        self.screen.blit(label, (cx - label.get_width() // 2, cy - radius - 30))

        # Grab status
        if wheel.grabbed:
            status = self.font_tiny.render("● GRABBED", True, INDICATOR_GREEN)
        else:
            status = self.font_tiny.render("○ READY", True, TEXT_DIM)
        self.screen.blit(status, (cx - status.get_width() // 2, cy + radius + 35))

    def draw_knob(self, cx: int, cy: int, radius: int, knob: Knob):
        """Draw a rotary knob at the given position."""
        # Background
        pygame.draw.circle(self.screen, KNOB_BG, (cx, cy), radius)

        # Arc showing value (270 degree sweep)
        start_angle = math.radians(135)  # Bottom-left
        sweep = math.radians(270 * knob.smoothed_value)

        # Draw background arc
        for a in range(0, 270, 2):
            angle = math.radians(135 + a)
            x = cx + math.cos(angle) * (radius - 4)
            y = cy + math.sin(angle) * (radius - 4)
            pygame.draw.circle(self.screen, KNOB_RING, (int(x), int(y)), 2)

        # Draw value arc
        value_degrees = int(270 * knob.smoothed_value)
        for a in range(0, value_degrees, 2):
            angle = math.radians(135 + a)
            x = cx + math.cos(angle) * (radius - 4)
            y = cy + math.sin(angle) * (radius - 4)
            pygame.draw.circle(self.screen, knob.color, (int(x), int(y)), 3)

        # Pointer line
        pointer_angle = math.radians(135 + 270 * knob.smoothed_value)
        px = cx + math.cos(pointer_angle) * (radius * 0.6)
        py = cy + math.sin(pointer_angle) * (radius * 0.6)
        pygame.draw.line(self.screen, ACCENT_WHITE, (cx, cy), (int(px), int(py)), 3)

        # Center dot
        pygame.draw.circle(self.screen, knob.color if knob.grabbed else ACCENT_DIM, (cx, cy), 5)

        # Label
        label = self.font_small.render(knob.label, True, knob.color)
        self.screen.blit(label, (cx - label.get_width() // 2, cy + radius + 10))

        # Value display
        val_text = self.font_tiny.render(f"{int(knob.smoothed_value * 100)}%", True, TEXT_DIM)
        self.screen.blit(val_text, (cx - val_text.get_width() // 2, cy + radius + 28))

        # Grab indicator
        if knob.grabbed:
            pygame.draw.circle(self.screen, INDICATOR_GREEN, (cx + radius + 10, cy), 4)

    def draw_webcam_feed(self, frame: np.ndarray, x: int, y: int, w: int, h: int):
        """Draw the webcam feed onto the Pygame surface."""
        if frame is None:
            return
        # Resize and convert
        frame_resized = cv2.resize(frame, (w, h))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        # Rotate/flip for pygame
        surf = pygame.surfarray.make_surface(np.rot90(frame_rgb, -1))
        surf = pygame.transform.flip(surf, True, False)

        # Border
        pygame.draw.rect(self.screen, WEBCAM_BORDER, (x - 2, y - 2, w + 4, h + 4), 2)
        self.screen.blit(surf, (x, y))

        # Label
        cam_label = self.font_tiny.render("WEBCAM TRACKING", True, TEXT_DIM)
        self.screen.blit(cam_label, (x, y - 16))

    def draw_hand_indicators(self, left_detected: bool, right_detected: bool,
                              left_gesture: str, right_gesture: str):
        """Draw hand detection status."""
        y_pos = WINDOW_HEIGHT - 45

        # Left hand
        color = INDICATOR_GREEN if left_detected else INDICATOR_RED
        pygame.draw.circle(self.screen, color, (30, y_pos), 6)
        txt = self.font_small.render(f"L: {left_gesture}" if left_detected else "L: ---", True, color)
        self.screen.blit(txt, (42, y_pos - 7))

        # Right hand
        color = INDICATOR_GREEN if right_detected else INDICATOR_RED
        pygame.draw.circle(self.screen, color, (WINDOW_WIDTH - 180, y_pos), 6)
        txt = self.font_small.render(f"R: {right_gesture}" if right_detected else "R: ---", True, color)
        self.screen.blit(txt, (WINDOW_WIDTH - 166, y_pos - 7))

    def draw_title_bar(self):
        """Draw the top title bar."""
        title = self.font_large.render("HAND DJ", True, ACCENT_WHITE)
        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 12))

        subtitle = self.font_tiny.render("AIR CONTROLLER v0.1", True, TEXT_DIM)
        self.screen.blit(subtitle, (WINDOW_WIDTH // 2 - subtitle.get_width() // 2, 44))

        # Divider
        pygame.draw.line(self.screen, ACCENT_DIM, (20, 65), (WINDOW_WIDTH - 20, 65), 1)

    def draw_instructions(self):
        """Draw control instructions."""
        instructions = [
            "OPEN PALM over wheel = Spin jogwheel, orbit hand around center  |  Release to carry momentum",
            "PINCH over knob = Grab knob, rotate wrist to turn  |  POINT = no action",
            "Left side of screen = Deck A  |  Right side = Deck B",
            "Press Q or ESC to quit",
        ]
        y = WINDOW_HEIGHT - 120
        for line in instructions:
            txt = self.font_tiny.render(line, True, TEXT_DIM)
            self.screen.blit(txt, (WINDOW_WIDTH // 2 - txt.get_width() // 2, y))
            y += 16


# ─── Main Application ───────────────────────────────────────────────────────

class HandDJApp:
    """Main application tying everything together."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Hand DJ Controller")
        self.clock = pygame.time.Clock()

        # Components
        self.hand_analyzer = HandAnalyzer()
        self.renderer = DJRenderer(self.screen)

        # DJ Controls
        self.wheel_a = JogWheel(deck_name="A", color=ACCENT_CYAN)
        self.wheel_b = JogWheel(deck_name="B", color=ACCENT_MAGENTA)
        self.knob_a = Knob(label="FILTER A", color=ACCENT_CYAN)
        self.knob_b = Knob(label="FILTER B", color=ACCENT_MAGENTA)

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not self.cap.isOpened():
            print("ERROR: Cannot open webcam. Check your camera connection.")
            sys.exit(1)

        self.last_frame = None
        self.running = True

        # Layout positions (will be computed)
        self.layout = self._compute_layout()

    def _compute_layout(self) -> dict:
        """Compute positions for all UI elements."""
        wheel_radius = 110
        knob_radius = 35
        deck_y = 330
        knob_y = 550

        cx = WINDOW_WIDTH // 2
        return {
            "wheel_a": (220, deck_y, wheel_radius),
            "wheel_b": (WINDOW_WIDTH - 220, deck_y, wheel_radius),
            "knob_a": (cx - 70, knob_y, knob_radius),
            "knob_b": (cx + 70, knob_y, knob_radius),
        }

    def _hand_in_zone(self, hx: float, hy: float, zone_center: Tuple[float, float]) -> bool:
        """Check if hand position is within an interaction zone."""
        zx, zy = zone_center
        dist = math.sqrt((hx - zx)**2 + (hy - zy)**2)
        return dist < GRAB_RADIUS

    def _process_hands(self, results):
        """Process detected hands and update DJ controls."""
        left_detected = False
        right_detected = False
        left_gesture = "---"
        right_gesture = "---"

        if not results.multi_hand_landmarks:
            self.wheel_a.grabbed = False
            self.wheel_b.grabbed = False
            self.knob_a.grabbed = False
            self.knob_b.grabbed = False
            return left_detected, right_detected, left_gesture, right_gesture

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            cx, cy = self.hand_analyzer.get_hand_center(hand_landmarks)
            pinch     = self.hand_analyzer.is_pinch(hand_landmarks)
            pointing  = self.hand_analyzer.is_pointing(hand_landmarks)
            open_palm = self.hand_analyzer.is_open_palm(hand_landmarks) and not pinch and not pointing
            rotation  = self.hand_analyzer.get_hand_rotation(hand_landmarks)

            # Map normalized hand pos to screen pixels
            hsx = cx * WINDOW_WIDTH
            hsy = cy * WINDOW_HEIGHT

            # Deck selection by screen position
            if cx <= 0.5:
                left_detected = True
                wheel = self.wheel_a
                knob  = self.knob_a
                wx, wy, _ = self.layout["wheel_a"]
                kx, ky, _ = self.layout["knob_a"]
                left_gesture = "PALM" if open_palm else ("PINCH" if pinch else ("POINT" if pointing else "---"))
            else:
                right_detected = True
                wheel = self.wheel_b
                knob  = self.knob_b
                wx, wy, _ = self.layout["wheel_b"]
                kx, ky, _ = self.layout["knob_b"]
                right_gesture = "PALM" if open_palm else ("PINCH" if pinch else ("POINT" if pointing else "---"))

            in_wheel_zone = math.sqrt((hsx - wx)**2 + (hsy - wy)**2) < WHEEL_ZONE_PX
            in_knob_zone  = math.sqrt((hsx - kx)**2 + (hsy - ky)**2) < KNOB_ZONE_PX

            if open_palm and in_wheel_zone:
                # Angular tracking: angle of hand relative to wheel center → 360° spin
                hand_angle = math.degrees(math.atan2(hsy - wy, hsx - wx))
                if not wheel.grabbed:
                    wheel.grabbed = True
                    wheel.last_hand_angle = hand_angle
                else:
                    delta = hand_angle - wheel.last_hand_angle
                    if delta > 180:  delta -= 360
                    elif delta < -180: delta += 360
                    wheel.angle += delta
                    wheel.velocity = delta * FPS  # carry momentum on release
                    wheel.last_hand_angle = hand_angle
                knob.grabbed = False

            elif pinch and in_knob_zone:
                # Pinch + hand rotation → knob turn
                if not knob.grabbed:
                    knob.grabbed = True
                    knob.last_rotation = rotation
                else:
                    delta_rot = rotation - knob.last_rotation
                    if delta_rot > 180:  delta_rot -= 360
                    elif delta_rot < -180: delta_rot += 360
                    knob.value = max(0.0, min(1.0, knob.value + delta_rot / 270.0))
                    knob.last_rotation = rotation
                wheel.grabbed = False

            else:
                wheel.grabbed = False
                knob.grabbed = False

        return left_detected, right_detected, left_gesture, right_gesture

    def run(self):
        """Main application loop."""
        print("\n" + "="*50)
        print("  HAND DJ CONTROLLER")
        print("  Point your webcam at your hands")
        print("  OPEN PALM to grab jogwheels, drag left/right to spin")
        print("  POINT (index finger) to grab knobs")
        print("  Press Q or ESC to quit")
        print("="*50 + "\n")

        while self.running:
            dt = self.clock.tick(FPS) / 1000.0

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        self.running = False

            # Capture camera frame
            ret, frame = self.cap.read()
            if ret:
                # Mirror the frame for hand detection
                frame = cv2.flip(frame, 1)
                # Process hands
                results = self.hand_analyzer.process(frame)
                left_det, right_det, left_gest, right_gest = self._process_hands(results)
                # Draw landmarks on frame
                frame = self.hand_analyzer.draw_landmarks_on_frame(frame, results)
                self.last_frame = frame
            else:
                left_det, right_det = False, False
                left_gest, right_gest = "---", "---"

            # Update physics
            self.wheel_a.update(dt)
            self.wheel_b.update(dt)
            self.knob_a.update()
            self.knob_b.update()

            # ── Render ──
            self.renderer.draw_background(self.last_frame)
            self.renderer.draw_title_bar()

            # Jogwheels
            wa = self.layout["wheel_a"]
            wb = self.layout["wheel_b"]
            self.renderer.draw_jogwheel(*wa, self.wheel_a)
            self.renderer.draw_jogwheel(*wb, self.wheel_b)

            # Knobs
            ka = self.layout["knob_a"]
            kb = self.layout["knob_b"]
            self.renderer.draw_knob(*ka, self.knob_a)
            self.renderer.draw_knob(*kb, self.knob_b)

            # Hand status
            self.renderer.draw_hand_indicators(left_det, right_det, left_gest, right_gest)

            # Instructions
            self.renderer.draw_instructions()

            pygame.display.flip()

        # Cleanup
        self.cap.release()
        pygame.quit()
        cv2.destroyAllWindows()


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = HandDJApp()
    app.run()
