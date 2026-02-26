import numpy as np
import cv2

class ViewTransformer():
    def __init__(self):
        court_width, court_length = 68, 23.32
        self.pixel_verticies = np.array([[110,1035],[265, 275], [910,260],[1640, 915]], dtype=np.float32)
        self.target_verticies = np.array([[0,court_width],[0,0],[court_length,0],[court_length,court_width]], dtype=np.float32)
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_verticies, self.target_verticies)

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        if cv2.pointPolygonTest(self.pixel_verticies, p, False) < 0: return None
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        return transform_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info.get('position_adjusted')
                    if position is not None:
                        pos_transformed = self.transform_point(np.array(position))
                        if pos_transformed is not None:
                            track_info['position_transformed'] = pos_transformed.squeeze().tolist() 