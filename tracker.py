#!/usr/bin/env python3
"""
Real-time object tracker.

Opens your webcam, lets you select an object (click on it or drag a box
around it), and then follows it with a square in real time.

Controls
--------
  Left click            : place a tracking box centered on the click point
  Left click + drag     : draw a custom tracking box
  +  /  -               : grow / shrink the default click box
  r                     : reset (stop tracking, pick a new object)
  c                     : cycle tracker algorithm (KCF / MOSSE / CSRT)
  q  or  ESC            : quit
"""

import argparse
import time

import cv2


# Trackers ordered from fastest to most accurate.
#   MOSSE : extremely fast, lower accuracy           -> best for "very fast"
#   KCF   : fast, good accuracy                       -> great all-rounder
#   CSRT  : slower, highest accuracy                  -> best for tricky objects
TRACKER_ORDER = ["KCF", "MOSSE", "CSRT"]


def create_tracker(name):
    """Create an OpenCV tracker by name across OpenCV versions."""
    name = name.upper()

    # OpenCV >= 4.5.1 exposes trackers directly; older ones use the legacy ns.
    factories = {
        "KCF": [
            getattr(cv2, "TrackerKCF_create", None),
            getattr(getattr(cv2, "legacy", None), "TrackerKCF_create", None),
        ],
        "MOSSE": [
            getattr(getattr(cv2, "legacy", None), "TrackerMOSSE_create", None),
            getattr(cv2, "TrackerMOSSE_create", None),
        ],
        "CSRT": [
            getattr(cv2, "TrackerCSRT_create", None),
            getattr(getattr(cv2, "legacy", None), "TrackerCSRT_create", None),
        ],
    }

    for factory in factories.get(name, []):
        if factory is not None:
            return factory()

    raise RuntimeError(
        f"Tracker '{name}' is not available in your OpenCV build. "
        "Install 'opencv-contrib-python' to get all trackers."
    )


class TrackerApp:
    def __init__(self, camera_index=0, tracker_name="CSRT", box_size=120,
                 width=None, height=None, mirror=True):
        self.camera_index = camera_index
        self.tracker_name = tracker_name.upper()
        self.box_size = box_size
        self.req_width = width
        self.req_height = height
        self.mirror = mirror

        self.tracker = None
        self.tracking = False
        self.bbox = None  # (x, y, w, h)

        # Template of the object, kept for automatic re-acquisition when the
        # tracker loses the lock.
        self.template = None
        self.template_size = None
        self.lost_frames = 0

        # Mouse drag state.
        self.drag_start = None
        self.drag_now = None
        self.dragging = False

        self.window = "Object Tracker"

    # -- mouse handling -------------------------------------------------
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_now = (x, y)
            self.dragging = True

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag_now = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            x0, y0 = self.drag_start
            dx, dy = abs(x - x0), abs(y - y0)

            if dx < 8 or dy < 8:
                # Treated as a click: build a box centered on the click point.
                s = self.box_size
                bbox = (x - s // 2, y - s // 2, s, s)
            else:
                # Treated as a drag: use the rectangle the user drew.
                bbox = (min(x0, x), min(y0, y), dx, dy)

            self.start_tracking(param, bbox)
            self.drag_start = None
            self.drag_now = None

    # -- tracking -------------------------------------------------------
    def start_tracking(self, frame, bbox, store_template=True):
        x, y, w, h = bbox
        h_img, w_img = frame.shape[:2]

        # Clamp the box to the frame.
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(10, min(w, w_img - x))
        h = max(10, min(h, h_img - y))
        bbox = (x, y, w, h)

        self.tracker = create_tracker(self.tracker_name)
        self.tracker.init(frame, bbox)
        self.bbox = bbox
        self.tracking = True
        self.lost_frames = 0

        # Remember what the object looks like (only on a fresh user selection),
        # so we can re-find it later if the tracker fails.
        if store_template:
            roi = frame[y:y + h, x:x + w]
            if roi.size > 0:
                self.template = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                self.template_size = (w, h)

    def reacquire(self, frame):
        """Search the whole frame for the saved template and re-lock on it."""
        if self.template is None:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tw, th = self.template_size
        if gray.shape[0] < th or gray.shape[1] < tw:
            return False

        res = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        # 0.45 is a forgiving threshold; raise it if it re-locks onto the
        # wrong thing, lower it if it fails to re-find the object.
        if max_val >= 0.45:
            self.start_tracking(frame, (max_loc[0], max_loc[1], tw, th),
                                store_template=False)
            return True
        return False

    def reset(self):
        self.tracker = None
        self.tracking = False
        self.bbox = None
        self.template = None
        self.template_size = None
        self.lost_frames = 0

    # -- main loop ------------------------------------------------------
    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if self.req_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
        if self.req_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self.camera_index}. "
                "Check that a webcam is connected and not used by another app."
            )

        cv2.namedWindow(self.window)

        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Could not read a frame from the camera.")
        if self.mirror:
            frame = cv2.flip(frame, 1)
        cv2.setMouseCallback(self.window, self.on_mouse, frame)

        prev_t = time.time()
        fps = 0.0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if self.mirror:
                frame = cv2.flip(frame, 1)

            # Keep the latest frame available to the mouse callback.
            cv2.setMouseCallback(self.window, self.on_mouse, frame)

            if self.tracking and self.tracker is not None:
                ok, box = self.tracker.update(frame)
                if ok:
                    self.lost_frames = 0
                    self.bbox = tuple(int(v) for v in box)
                    x, y, w, h = self.bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)
                    cx, cy = x + w // 2, y + h // 2
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                else:
                    # Tracker lost it — try to automatically re-find it.
                    self.lost_frames += 1
                    if self.reacquire(frame):
                        x, y, w, h = self.bbox
                        cv2.rectangle(frame, (x, y), (x + w, y + h),
                                      (0, 200, 255), 2)
                    else:
                        cv2.putText(frame, "Searching... (press r to reselect)",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 0, 255), 2)

            # Draw the rubber-band rectangle while dragging.
            if self.dragging and self.drag_start and self.drag_now:
                cv2.rectangle(frame, self.drag_start, self.drag_now,
                              (255, 200, 0), 1)

            # FPS (exponential moving average).
            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            status = "TRACKING" if self.tracking else "Click an object"
            cv2.putText(frame,
                        f"{self.tracker_name} | {status} | {fps:4.1f} FPS",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)

            cv2.imshow(self.window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            elif key == ord("r"):
                self.reset()
            elif key in (ord("+"), ord("=")):
                self.box_size = min(self.box_size + 20, 600)
            elif key in (ord("-"), ord("_")):
                self.box_size = max(self.box_size - 20, 20)
            elif key == ord("c"):
                idx = TRACKER_ORDER.index(self.tracker_name) \
                    if self.tracker_name in TRACKER_ORDER else 0
                self.tracker_name = TRACKER_ORDER[(idx + 1) % len(TRACKER_ORDER)]
                self.reset()
            elif key == ord("m"):
                self.mirror = not self.mirror

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-time object tracker.")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0).")
    parser.add_argument("--tracker", default="CSRT",
                        choices=["KCF", "MOSSE", "CSRT"],
                        help="Tracking algorithm (default: CSRT, most robust).")
    parser.add_argument("--no-mirror", action="store_true",
                        help="Disable the mirror (flipped) camera view.")
    parser.add_argument("--box-size", type=int, default=120,
                        help="Default box size for a single click (px).")
    parser.add_argument("--width", type=int, default=None,
                        help="Requested camera capture width.")
    parser.add_argument("--height", type=int, default=None,
                        help="Requested camera capture height.")
    args = parser.parse_args()

    app = TrackerApp(camera_index=args.camera, tracker_name=args.tracker,
                     box_size=args.box_size, width=args.width,
                     height=args.height, mirror=not args.no_mirror)
    app.run()


if __name__ == "__main__":
    main()
