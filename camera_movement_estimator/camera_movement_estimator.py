import pickle
import numpy as np
import cv2
import os
import sys
sys.path.append('./')
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator:
    def __init__(self, frame):
        self.minimum_distance = 0.5 
        # Track total movement across the whole video globally
        self.accumulated_x = 0.0
        self.accumulated_y = 0.0

        self.lk_params = dict(
            winSize = (15, 15), 
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        
        # High-confidence areas: stands and sidelines
        mask_features[0:300, :] = 1 
        mask_features[:, 0:150] = 1
        mask_features[:, -150:] = 1
        
        self.features = dict(
            maxCorners = 200,
            qualityLevel = 0.3,
            minDistance = 7,
            blockSize = 7,
            mask = mask_features
        )

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # FIX: Every frame in this batch should at least start with the previous total
        camera_movement = [[self.accumulated_x, self.accumulated_y] for _ in range(len(frames))]
        
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
        
        if old_features is None:
            return camera_movement

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            if new_features is not None and status is not None:
                good_new = new_features[status == 1]
                good_old = old_features[status == 1]

                if len(good_new) > 0:
                    good_new = good_new.reshape(-1, 2)
                    good_old = good_old.reshape(-1, 2)
                    
                    diffs = good_new - good_old
                    avg_move_x = np.mean(diffs[:, 0])
                    avg_move_y = np.mean(diffs[:, 1])

                    if abs(avg_move_x) > self.minimum_distance or abs(avg_move_y) > self.minimum_distance:
                        self.accumulated_x += avg_move_x
                        self.accumulated_y += avg_move_y
                        # Reset features to avoid tracking "stale" points as they move out of mask
                        old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                    else:
                        old_features = good_new.reshape(-1, 1, 2)
                
                # Store the updated accumulation
                camera_movement[frame_num] = [float(self.accumulated_x), float(self.accumulated_y)]

            old_gray = frame_gray.copy()

        # Note: In batch mode, we don't save the stub here because it would 
        # overwrite itself every 30 frames with only partial data.
        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []
        for i, frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()
            # Slightly wider rectangle to fit "Total Cam X: -000.00"
            cv2.rectangle(overlay, (10, 10), (450, 110), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            x, y = camera_movement_per_frame[i]
            cv2.putText(frame, f"Total Cam X: {x:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            cv2.putText(frame, f"Total Cam Y: {y:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            output_frames.append(frame)
        return output_frames