import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
from utils import measure_distance, measure_xy_distance

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.38
FONT_THICK = 1
COL_WHITE  = (255, 255, 255)
COL_BLACK  = (  0,   0,   0)
COL_DARK   = ( 20,  20,  20)
COL_ACCENT = (100, 220, 255)   # soft cyan accent


def _alpha_rect(frame, x1, y1, x2, y2, color, alpha=0.60):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


class CameraMovementEstimator:
    def __init__(self, frame):
        self.minimum_distance = 5
        self.lk_params = dict(
            winSize  = (15, 15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        10, 0.03)
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        mask[:, 0:20]       = 1
        mask[:, 900:1050]   = 1
        self.features = dict(
            maxCorners  = 100,
            qualityLevel= 0.3,
            minDistance = 3,
            blockSize   = 7,
            mask        = mask
        )

    def add_adjust_positions_to_tracks(self, tracks,
                                       camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    pos = track_info['position']
                    cm  = camera_movement_per_frame[frame_num]
                    tracks[object][frame_num][track_id][
                        'position_adjusted'] = (pos[0] - cm[0],
                                                pos[1] - cm[1])

    def get_camera_movement(self, frames,
                            read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0, 0]] * len(frames)
        old_gray        = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features    = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            gray         = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, gray, old_features, None, **self.lk_params)

            max_dist, cx, cy = 0, 0, 0
            for new, old in zip(new_features, old_features):
                np_ = new.ravel()
                op_ = old.ravel()
                d   = measure_distance(np_, op_)
                if d > max_dist:
                    max_dist = d
                    cx, cy   = measure_xy_distance(op_, np_)

            if max_dist > self.minimum_distance:
                camera_movement[frame_num] = [cx, cy]
                old_features = cv2.goodFeaturesToTrack(
                    gray, **self.features)
            old_gray = gray.copy()

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)
        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        """
        Compact top-left HUD panel showing camera drift.
        Only shown when movement exceeds threshold to reduce noise.
        """
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            cx, cy = camera_movement_per_frame[frame_num]

            # panel: 210 wide, 52 tall, top-left with margin
            px1, py1 = 12, 12
            px2, py2 = 222, 64

            _alpha_rect(frame, px1, py1, px2, py2, COL_DARK, alpha=0.65)
            # thin accent border
            cv2.rectangle(frame, (px1, py1), (px2, py2),
                          COL_ACCENT, 1, cv2.LINE_AA)

            # header label
            cv2.putText(frame, "CAM DRIFT",
                        (px1 + 8, py1 + 14),
                        FONT, FONT_SCALE, COL_ACCENT,
                        FONT_THICK, cv2.LINE_AA)

            # X / Y values
            x_col = COL_WHITE if abs(cx) < 2 else (80, 180, 255)
            y_col = COL_WHITE if abs(cy) < 2 else (80, 180, 255)

            cv2.putText(frame, f"X {cx:+.1f}  Y {cy:+.1f}",
                        (px1 + 8, py1 + 38),
                        FONT, FONT_SCALE, COL_WHITE,
                        FONT_THICK, cv2.LINE_AA)

            output_frames.append(frame)
        return output_frames