from .tracker import Tracker

def __init__(self, model_path):
    self.model   = YOLO(model_path)
    self.tracker = sv.ByteTrack(
        track_activation_threshold=0.35,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=24
    )