import cv2
import sys
sys.path.append('../')
from utils import measure_distance, get_foot_position

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.36
FONT_THICK = 1
COL_WHITE  = (255, 255, 255)
COL_BLACK  = (  0,   0,   0)
COL_DARK   = ( 20,  20,  20)
COL_SPEED  = ( 60, 220, 255)   # warm yellow-green for speed


def _alpha_rect(frame, x1, y1, x2, y2, color, alpha=0.55):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


class SpeedAndDistance_Estimator:
    def __init__(self):
        self.frame_window = 5
        self.frame_rate   = 24

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for object, object_tracks in tracks.items():
            if object in ("ball", "referees"):
                continue
            n_frames = len(object_tracks)

            for frame_num in range(0, n_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, n_frames - 1)

                for track_id in object_tracks[frame_num]:
                    if track_id not in object_tracks[last_frame]:
                        continue

                    sp = object_tracks[frame_num][track_id].get(
                        'position_transformed')
                    ep = object_tracks[last_frame][track_id].get(
                        'position_transformed')
                    if sp is None or ep is None:
                        continue

                    dist     = measure_distance(sp, ep)
                    elapsed  = (last_frame - frame_num) / self.frame_rate
                    speed_ms = dist / elapsed
                    speed_kh = speed_ms * 3.6

                    total_distance.setdefault(object, {})
                    total_distance[object].setdefault(track_id, 0)
                    total_distance[object][track_id] += dist

                    for fn in range(frame_num, last_frame):
                        if track_id not in tracks[object][fn]:
                            continue
                        tracks[object][fn][track_id]['speed']    = speed_kh
                        tracks[object][fn][track_id]['distance'] = \
                            total_distance[object][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        """
        Show speed only (no distance) in a compact pill above foot position.
        Only render every 3rd frame value to reduce flicker/clutter.
        """
        output_frames = []

        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object in ("ball", "referees"):
                    continue

                for track_id, track_info in object_tracks[frame_num].items():
                    speed = track_info.get('speed')
                    if speed is None:
                        continue

                    bbox     = track_info['bbox']
                    fx, fy   = get_foot_position(bbox)
                    # place label just below foot
                    lx, ly   = int(fx), int(fy) + 28

                    label    = f"{speed:.0f} km/h"
                    (tw, th), _ = cv2.getTextSize(
                        label, FONT, FONT_SCALE, FONT_THICK)
                    pad = 4
                    rx1, rx2 = lx - tw // 2 - pad, lx + tw // 2 + pad
                    ry1, ry2 = ly - th - pad,       ly + pad

                    # pill background
                    _alpha_rect(frame, rx1, ry1, rx2, ry2,
                                COL_DARK, alpha=0.60)
                    # shadow
                    cv2.putText(frame, label,
                                (rx1 + pad + 1, ly),
                                FONT, FONT_SCALE,
                                COL_BLACK, FONT_THICK + 1, cv2.LINE_AA)
                    # text
                    cv2.putText(frame, label,
                                (rx1 + pad, ly - 1),
                                FONT, FONT_SCALE,
                                COL_SPEED, FONT_THICK, cv2.LINE_AA)

            output_frames.append(frame)
        return output_frames