"""
Hand Gesture Control - control your laptop with your webcam and your hand.

Install:  pip install opencv-python mediapipe pyautogui numpy
Run:      python hand_gesture_control.py
Quit:     press  q  or  Esc  in the camera window
          (or push the real mouse into a screen corner: pyautogui fail-safe)

GESTURES  (hold your hand upright in front of the webcam)
  Index finger up only ............ move the cursor
  Thumb + index pinch ............. left click  (keep pinching to drag)
  Thumb + middle pinch ............ right click
  Index + middle up (peace sign) .. scroll: raise hand = up, lower hand = down
  Index + middle + ring up ........ volume: raise hand = up, lower hand = down
  Pinky up (others folded, thumb
  optional - the 🤙 shaka sign) .. video seek: move hand RIGHT = forward,
                                    LEFT = backward (works in YouTube etc.)
  Open palm ....................... idle (nothing happens)
  Fist held for 1.5 seconds ....... pause / resume everything
"""

import math
import subprocess
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# ============================== SETTINGS ==============================
CAMERA_INDEX = 0            # try 1 if the wrong camera opens
CAM_W, CAM_H = 640, 480

FRAME_MARGIN = 100          # px border of the camera image that is ignored,
                            # so you can reach screen edges without stretching
SMOOTHING = 5               # higher = smoother but laggier cursor
MOVE_DEADZONE = 1           # px: ignore tiny jitter

PINCH_ON = 0.28             # pinch distance / palm size needed to START a pinch
PINCH_OFF = 0.42            # distance needed to RELEASE it (hysteresis = no flicker)

SCROLL_DEADZONE = 25        # px the hand must move from its start point
SCROLL_DIVISOR = 40         # bigger = slower scrolling
SCROLL_MAX = 10
SCROLL_UNIT = 60 if sys.platform.startswith("win") else 1  # Windows scroll units are tiny

VOLUME_DEADZONE = 30
VOLUME_INTERVAL = 0.12      # seconds between volume steps

SEEK_FORWARD_KEY = "right"  # YouTube: right arrow = +5s. Use "l" for +10s
SEEK_BACK_KEY = "left"      # YouTube: left arrow  = -5s. Use "j" for -10s
SEEK_DEADZONE = 40          # px the hand must move sideways before seeking starts
SEEK_FAST_OFFSET = 120      # move further than this -> seek faster
SEEK_SLOW_INTERVAL = 0.40   # seconds between key presses (gentle seek)
SEEK_FAST_INTERVAL = 0.12   # seconds between key presses (fast seek)

RIGHT_CLICK_COOLDOWN = 0.8
PAUSE_HOLD_SECONDS = 1.5

pyautogui.FAILSAFE = True   # slam the mouse into a screen corner to abort
pyautogui.PAUSE = 0         # no built-in delay after each action
# ======================================================================

# MediaPipe hand landmark ids
WRIST, THUMB_IP, THUMB_TIP = 0, 3, 4
INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, PINKY_MCP = 8, 9, 12, 17
FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))  # (tip, pip) of index..pinky


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def fingers_up(pts):
    """Return [thumb, index, middle, ring, pinky] -> True if extended."""
    thumb = dist(pts[THUMB_TIP], pts[PINKY_MCP]) > dist(pts[THUMB_IP], pts[PINKY_MCP])
    others = [dist(pts[tip], pts[WRIST]) > dist(pts[pip], pts[WRIST]) for tip, pip in FINGERS]
    return [thumb] + others


def change_volume(direction):
    """direction: +1 = louder, -1 = quieter."""
    if sys.platform == "darwin":
        step = 6 * direction
        script = f"set volume output volume ((output volume of (get volume settings)) + {step})"
        subprocess.Popen(["osascript", "-e", script])
    else:
        pyautogui.press("volumeup" if direction > 0 else "volumedown")


class GestureController:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        self.cur_x, self.cur_y = pyautogui.position()
        self.mode = None
        self.paused = False
        self.mouse_down = False
        self.anchor_y = None
        self.last_right_click = 0.0
        self.last_volume = 0.0
        self.anchor_x = None
        self.last_seek = 0.0
        self.pinching = False
        self.fist_start = None
        self.fist_used = False

    # ----------------------------- actions -----------------------------
    def set_button(self, down):
        if down and not self.mouse_down:
            pyautogui.mouseDown()
        elif not down and self.mouse_down:
            pyautogui.mouseUp()
        self.mouse_down = down

    def move_cursor(self, pt, w, h):
        x = np.interp(pt[0], (FRAME_MARGIN, w - FRAME_MARGIN), (0, self.screen_w))
        y = np.interp(pt[1], (FRAME_MARGIN, h - FRAME_MARGIN), (0, self.screen_h))
        # stay 2px away from the corners so pyautogui's fail-safe isn't tripped by accident
        x = float(np.clip(x, 2, self.screen_w - 3))
        y = float(np.clip(y, 2, self.screen_h - 3))
        nx = self.cur_x + (x - self.cur_x) / SMOOTHING
        ny = self.cur_y + (y - self.cur_y) / SMOOTHING
        if math.hypot(nx - self.cur_x, ny - self.cur_y) >= MOVE_DEADZONE:
            self.cur_x, self.cur_y = nx, ny
            pyautogui.moveTo(nx, ny)

    def vertical_offset(self, y):
        """How far the hand moved up (+) or down (-) since the gesture began."""
        if self.anchor_y is None:
            self.anchor_y = y
        return self.anchor_y - y

    def scroll(self, y):
        offset = self.vertical_offset(y)
        if abs(offset) > SCROLL_DEADZONE:
            speed = min(SCROLL_MAX, 1 + (abs(offset) - SCROLL_DEADZONE) / SCROLL_DIVISOR)
            pyautogui.scroll(int(math.copysign(speed, offset) * SCROLL_UNIT))

    def volume(self, y, now):
        offset = self.vertical_offset(y)
        if abs(offset) > VOLUME_DEADZONE and now - self.last_volume > VOLUME_INTERVAL:
            self.last_volume = now
            change_volume(1 if offset > 0 else -1)

    def seek(self, x, now):
        if self.anchor_x is None:
            self.anchor_x = x
        offset = x - self.anchor_x          # image is mirrored: your right = positive
        if abs(offset) < SEEK_DEADZONE:
            return
        fast = abs(offset) > SEEK_FAST_OFFSET
        interval = SEEK_FAST_INTERVAL if fast else SEEK_SLOW_INTERVAL
        if now - self.last_seek > interval:
            self.last_seek = now
            pyautogui.press(SEEK_FORWARD_KEY if offset > 0 else SEEK_BACK_KEY)

    # ------------------------------ logic ------------------------------
    def update(self, pts, w, h):
        """Look at one hand, perform the matching action, return a label."""
        now = time.time()
        palm = dist(pts[WRIST], pts[MIDDLE_MCP]) or 1.0   # hand-size ruler
        _, index, middle, ring, pinky = fingers_up(pts)
        d_index = dist(pts[THUMB_TIP], pts[INDEX_TIP]) / palm
        d_middle = dist(pts[THUMB_TIP], pts[MIDDLE_TIP]) / palm

        # pinch with hysteresis (starts at PINCH_ON, ends at PINCH_OFF)
        self.pinching = d_index < (PINCH_OFF if self.pinching else PINCH_ON)

        # ---- fist: hold to pause / resume ----
        fist = not (index or middle or ring or pinky) and not self.pinching
        if fist:
            self.set_button(False)
            self.mode, self.anchor_y = "FIST", None
            if self.fist_start is None:
                self.fist_start = now
            elif not self.fist_used and now - self.fist_start >= PAUSE_HOLD_SECONDS:
                self.paused = not self.paused
                self.fist_used = True
            return "FIST (hold to pause/resume)"
        self.fist_start, self.fist_used = None, False

        if self.paused:
            self.set_button(False)
            self.mode = "PAUSED"
            return "PAUSED (hold a fist to resume)"

        # ---- decide the mode ----
        if self.pinching:
            mode = "CLICK / DRAG"
        elif d_middle < PINCH_ON:
            mode = "RIGHT CLICK"
        elif index and not (middle or ring or pinky):
            mode = "MOVE"
        elif index and middle and not (ring or pinky):
            mode = "SCROLL"
        elif index and middle and ring and not pinky:
            mode = "VOLUME"
        elif pinky and not (index or middle or ring):
            mode = "SEEK"
        else:
            mode = "IDLE"

        if mode != self.mode:                       # gesture just changed
            self.anchor_y = None
            self.anchor_x = None
            if mode == "RIGHT CLICK" and now - self.last_right_click > RIGHT_CLICK_COOLDOWN:
                pyautogui.rightClick()
                self.last_right_click = now
            self.mode = mode

        # ---- act ----
        self.set_button(mode == "CLICK / DRAG")
        if mode in ("MOVE", "CLICK / DRAG"):
            self.move_cursor(pts[INDEX_TIP], w, h)
        elif mode == "SCROLL":
            self.scroll((pts[INDEX_TIP][1] + pts[MIDDLE_TIP][1]) / 2)
        elif mode == "VOLUME":
            self.volume((pts[INDEX_TIP][1] + pts[MIDDLE_TIP][1]) / 2, now)
        elif mode == "SEEK":
            self.seek(pts[MIDDLE_MCP][0], now)
        return mode

    def hand_lost(self):
        self.set_button(False)          # never leave the mouse button stuck down
        self.mode, self.anchor_y = None, None
        self.fist_start, self.fist_used = None, False


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        print("Could not open the camera. Try another CAMERA_INDEX.")
        return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )
    ctrl = GestureController()
    prev_time = time.time()
    print("Gesture control running. Press q or Esc in the camera window to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)                  # mirror: feels natural
            h, w = frame.shape[:2]
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            label = "No hand"
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                pts = [(lm.x * w, lm.y * h) for lm in hand.landmark]
                label = ctrl.update(pts, w, h)
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            else:
                ctrl.hand_lost()

            # ---- on-screen overlay ----
            cv2.rectangle(frame, (FRAME_MARGIN, FRAME_MARGIN),
                          (w - FRAME_MARGIN, h - FRAME_MARGIN), (255, 0, 255), 2)
            colour = (0, 0, 255) if ctrl.paused else (0, 255, 0)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
            now = time.time()
            cv2.putText(frame, f"FPS {1 / max(now - prev_time, 1e-6):.0f}", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            prev_time = now

            cv2.imshow("Hand Gesture Control", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    except pyautogui.FailSafeException:
        print("Fail-safe triggered (mouse hit a screen corner). Exiting.")
    finally:
        ctrl.hand_lost()
        cap.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    
