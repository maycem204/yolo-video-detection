import cv2
import torch
import gc
from ultralytics import YOLO
import supervision as sv
import numpy as np
from utils.bbox_utils import get_center_of_bbox, get_bbox_width, get_foot_position
import pandas as pd

class Tracker:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path).to(self.device)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object_type == 'ball':
                        positions = get_center_of_bbox(bbox)
                    else:
                        positions = get_foot_position(bbox)
                    tracks[object_type][frame_num][track_id]['position_adjusted'] = positions

    def interpolate_ball_positions(self, ball_positions):
        ball_bboxes = [x.get(1, {}).get('bbox', []) for x in ball_positions] 
        df_ball_positions = pd.DataFrame(ball_bboxes, columns=['x1','y1','x2','y2'])
        df_ball_positions = df_ball_positions.interpolate().bfill()
        return [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

    def detect_frames(self, frames):
        detections = []
        for frame in frames:
            results = self.model.predict(frame, conf=0.1, verbose=False, device=self.device)
            for res in results:
                detections.append(res.cpu())
            del results
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        return detections

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)
        tracks = {"players": [], "ball": [], "referees": []}

        for detection in detections:
            cls_names_inv = {v: k for k, v in detection.names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)
            tracks_output = self.tracker.update_with_detections(detection_supervision)

            p_f, r_f, b_f = {}, {}, {}
            for track in tracks_output:
                bbox, cls_id, tid = track[0].tolist(), int(track[3]), int(track[4])
                if cls_id == cls_names_inv.get('player'): p_f[tid] = {"bbox": bbox}
                elif cls_id == cls_names_inv.get('referee'): r_f[tid] = {"bbox": bbox}

            for det in detection_supervision:
                if int(det[3]) == cls_names_inv.get('ball'): b_f[1] = {"bbox": det[0].tolist()}

            tracks["players"].append(p_f)
            tracks["referees"].append(r_f)
            tracks["ball"].append(b_f)
        return tracks

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        history = team_ball_control[:frame_num + 1]
        t1, t2 = history.count(1), history.count(2)
        total = t1 + t2
        t1_pct = (t1 / total * 100) if total > 0 else 0
        t2_pct = (t2 / total * 100) if total > 0 else 0

        cv2.putText(frame, f'Team 1: {t1_pct:.1f}%', (1360, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f'Team 2: {t2_pct:.1f}%', (1360, 940), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control, start_frame):
        output_frames = []
        for i, frame in enumerate(video_frames):
            frame = frame.copy()
            frame = self.draw_team_ball_control(frame, start_frame + i, team_ball_control)

            # Draw Referees
            for tid, r in tracks["referees"][i].items():
                x_c, _ = get_center_of_bbox(r["bbox"])
                y2, w = int(r["bbox"][3]), get_bbox_width(r["bbox"])
                cv2.ellipse(frame, (x_c, y2), (int(w), int(0.35 * w)), 0, -45, 235, (255, 255, 255), 2)

            # Draw Players
            for tid, p in tracks["players"][i].items():
                color = p.get('team_color', (0, 0, 255))
                x_c, _ = get_center_of_bbox(p["bbox"])
                y2, w = int(p["bbox"][3]), get_bbox_width(p["bbox"])
                cv2.ellipse(frame, (x_c, y2), (int(w), int(0.35 * w)), 0, -45, 235, color, 2)
                
                if p.get('has_ball', False):
                    y_top = int(p["bbox"][1])
                    tri = np.array([[x_c, y_top], [x_c - 10, y_top - 20], [x_c + 10, y_top - 20]])
                    cv2.drawContours(frame, [tri], 0, (0,0,255), -1)

                rx1, ry1 = x_c - 20, y2 + 5
                cv2.rectangle(frame, (int(rx1), int(ry1)), (int(rx1 + 40), int(ry1 + 20)), color, -1)
                cv2.putText(frame, str(tid), (int(rx1 + 5), int(ry1 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Draw Ball
            for _, b in tracks["ball"][i].items():
                x, y = get_center_of_bbox(b["bbox"])[0], int(b["bbox"][1])
                pts = np.array([[x, y], [x-10, y-20], [x+10, y-20]])
                cv2.drawContours(frame, [pts], 0, (0, 255, 0), -1)

            output_frames.append(frame)
        return output_frames