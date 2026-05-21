from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

# ── Design tokens ────────────────────────────────────────────────
FONT          = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL    = 0.38
FONT_MED      = 0.52
FONT_THICK    = 1

# Team / object colours  (BGR)
COL_TEAM1     = (235, 100,  30)   # vivid orange
COL_TEAM2     = ( 30, 200,  80)   # vivid green
COL_REF       = ( 30, 220, 220)   # cyan
COL_BALL      = ( 20, 220, 255)   # yellow-green
COL_HAS_BALL  = (  0,  60, 255)   # red indicator
COL_WHITE     = (255, 255, 255)
COL_BLACK     = (  0,   0,   0)
COL_DARK      = ( 20,  20,  20)


def _alpha_rect(frame, x1, y1, x2, y2, color, alpha=0.55, radius=6):
    """Draw a filled rounded-ish rectangle with alpha blend."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # thin border
    cv2.rectangle(frame, (x1, y1), (x2, y2), COL_WHITE, 1, cv2.LINE_AA)


def _text_with_shadow(frame, text, pos, font_scale, color,
                      thickness=FONT_THICK):
    # shadow
    cv2.putText(frame, text,
                (pos[0]+1, pos[1]+1), FONT,
                font_scale, COL_BLACK, thickness + 1, cv2.LINE_AA)
    # main
    cv2.putText(frame, text, pos, FONT,
                font_scale, color, thickness, cv2.LINE_AA)


class Tracker:
    def __init__(self, model_path):
        self.model   = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    # ── position helpers ─────────────────────────────────────────
    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    position = (get_center_of_bbox(bbox)
                                if object == 'ball'
                                else get_foot_position(bbox))
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', [])
                         for x in ball_positions]
        df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df = df.interpolate().bfill()
        return [{1: {"bbox": x}} for x in df.to_numpy().tolist()]

    # ── detection / tracking ─────────────────────────────────────
    def detect_frames(self, frames):
        detections = []
        for i in range(0, len(frames), 20):
            detections += self.model.predict(frames[i:i+20], conf=0.35)  # was 0.1
        return detections

    def get_object_tracks(self, frames,
                          read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)
        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_names     = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            det_sv = sv.Detections.from_ultralytics(detection)

            for idx, class_id in enumerate(det_sv.class_id):
                if cls_names[class_id] == "goalkeeper":
                    det_sv.class_id[idx] = cls_names_inv["player"]

            det_tracked = self.tracker.update_with_detections(det_sv)
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for fd in det_tracked:
                bbox     = fd[0].tolist()
                cls_id   = fd[3]
                track_id = fd[4]
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for fd in det_sv:
                bbox   = fd[0].tolist()
                cls_id = fd[3]
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        return tracks

    # ── drawing primitives ───────────────────────────────────────
    def _draw_player_marker(self, frame, bbox, color, track_id=None):
        """Thin arc under player + small pill ID label."""
        y2       = int(bbox[3])
        x_center = int(get_center_of_bbox(bbox)[0])
        width    = int(get_bbox_width(bbox))

        # arc — thinner, anti-aliased
        cv2.ellipse(frame,
                    center=(x_center, y2),
                    axes=(width, int(0.30 * width)),
                    angle=0.0,
                    startAngle=-40, endAngle=220,
                    color=color, thickness=2,
                    lineType=cv2.LINE_AA)

        if track_id is not None:
            label   = str(track_id)
            (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SMALL, FONT_THICK)
            pad     = 5
            rx1     = x_center - tw // 2 - pad
            rx2     = x_center + tw // 2 + pad
            ry1     = y2 + 6
            ry2     = y2 + th + 6 + pad * 2

            # pill background
            _alpha_rect(frame, rx1, ry1, rx2, ry2, color, alpha=0.75)
            _text_with_shadow(frame, label,
                              (rx1 + pad, ry2 - pad - 1),
                              FONT_SMALL, COL_WHITE)
        return frame

    def _draw_ball_marker(self, frame, bbox, color):
        """Small downward-pointing triangle for ball."""
        y   = int(bbox[1])
        x   = int(get_center_of_bbox(bbox)[0])
        pts = np.array([[x, y + 4],
                        [x - 7, y - 12],
                        [x + 7, y - 12]], np.int32)
        cv2.drawContours(frame, [pts], 0, color,    cv2.FILLED)
        cv2.drawContours(frame, [pts], 0, COL_BLACK, 1)
        return frame

    def _draw_has_ball_indicator(self, frame, bbox):
        """Small red dot above player who has the ball."""
        x = int(get_center_of_bbox(bbox)[0])
        y = int(bbox[1]) - 8
        cv2.circle(frame, (x, y), 5, COL_HAS_BALL, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, COL_WHITE,    1,  cv2.LINE_AA)
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """Modern pill-shaped ball-control bar bottom-right."""
        h, w = frame.shape[:2]

        ctrl = team_ball_control[:frame_num + 1]
        n1   = (ctrl == 1).sum()
        n2   = (ctrl == 2).sum()
        tot  = n1 + n2 or 1
        p1, p2 = n1 / tot, n2 / tot

        # panel dimensions
        pw, ph = 320, 72
        px     = w - pw - 18
        py     = h - ph - 18

        # dark background panel
        _alpha_rect(frame, px, py, px + pw, py + ph,
                    COL_DARK, alpha=0.65)

        # header
        _text_with_shadow(frame, "BALL POSSESSION",
                          (px + 10, py + 18),
                          FONT_SMALL, COL_WHITE)

        # bar track
        bx1, bx2 = px + 10, px + pw - 10
        by1, by2 = py + 26, py + 44
        cv2.rectangle(frame, (bx1, by1), (bx2, by2),
                      (60, 60, 60), -1)

        # team 1 fill
        mid = int(bx1 + (bx2 - bx1) * p1)
        cv2.rectangle(frame, (bx1, by1), (mid, by2),
                      COL_TEAM1, -1)
        # team 2 fill
        cv2.rectangle(frame, (mid, by1), (bx2, by2),
                      COL_TEAM2, -1)
        # bar border
        cv2.rectangle(frame, (bx1, by1), (bx2, by2),
                      COL_WHITE, 1, cv2.LINE_AA)

        # percentage labels
        _text_with_shadow(frame, f"{p1*100:.0f}%",
                          (bx1 + 4, by2 + 16),
                          FONT_SMALL, COL_TEAM1)
        _text_with_shadow(frame, f"{p2*100:.0f}%",
                          (bx2 - 36, by2 + 16),
                          FONT_SMALL, COL_TEAM2)

        # team labels
        _text_with_shadow(frame, "T1",
                          (bx1 + 4, by1 - 4),
                          FONT_SMALL, COL_TEAM1)
        _text_with_shadow(frame, "T2",
                          (bx2 - 24, by1 - 4),
                          FONT_SMALL, COL_TEAM2)

        return frame

    # ── main annotation loop ─────────────────────────────────────
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict  = tracks["players"][frame_num]
            ball_dict    = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # referees — thin cyan arc, no ID
            for _, ref in referee_dict.items():
                self._draw_player_marker(frame, ref["bbox"], COL_REF)

            # players
            for track_id, player in player_dict.items():
                color = player.get("team_color", COL_TEAM1)
                self._draw_player_marker(frame, player["bbox"],
                                         color, track_id)
                if player.get('has_ball', False):
                    self._draw_has_ball_indicator(frame, player["bbox"])

            # ball
            for _, ball in ball_dict.items():
                self._draw_ball_marker(frame, ball["bbox"], COL_BALL)

            # possession bar
            frame = self.draw_team_ball_control(
                frame, frame_num, team_ball_control)

            output.append(frame)
        return output