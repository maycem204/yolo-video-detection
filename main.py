import traceback, cv2, os, gc
import numpy as np

from trackers import Tracker 
from utils.video_utils import read_video
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main():

    input_video_path = "input_videos/08fd33_4.mp4"
    output_video_path = "output_videos/output_video.avi"
    
    print("--- Initializing AI Models ---")

    tracker = Tracker('models/best.pt')
    team_assigner = TeamAssigner()
    player_assigner = PlayerBallAssigner()
    view_transformer = ViewTransformer()
    camera_movement_estimator = None

    # ✅ INITIALISATION SPEED & DISTANCE
    speed_and_distance_estimator = SpeedAndDistance_Estimator()

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not os.path.exists("output_videos"):
        os.makedirs("output_videos")

    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (width, height)
    )

    print(f"--- Starting Analysis: {input_video_path} ---")

    frames_generator = read_video(input_video_path)

    mini_batch = []
    frame_count = 0
    teams_initialized = False
    overall_team_ball_control = []

    try:

        for frame in frames_generator:

            mini_batch.append(frame)
            frame_count += 1

            if len(mini_batch) == 30:

                print(f"Processing frame range: {frame_count-29} to {frame_count}")

                # -----------------------
                # 1. TRACKING
                # -----------------------
                tracks = tracker.get_object_tracks(mini_batch)

                # -----------------------
                # 2. CAMERA MOVEMENT
                # -----------------------
                if camera_movement_estimator is None:
                    camera_movement_estimator = CameraMovementEstimator(mini_batch[0])

                cam_move = camera_movement_estimator.get_camera_movement(
                    mini_batch,
                    read_from_stub=False
                )

                # -----------------------
                # 3. POSITION & VIEW TRANSFORM
                # -----------------------
                tracker.add_position_to_tracks(tracks)

                view_transformer.add_transformed_position_to_tracks(tracks)

                tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

                # ✅ SPEED & DISTANCE CALCULATION
                speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

                # -----------------------
                # 4. TEAM INITIALIZATION
                # -----------------------
                if not teams_initialized:

                    for b_idx, f_tracks in enumerate(tracks['players']):

                        if len(f_tracks) > 5:

                            team_assigner.assign_team_color(
                                mini_batch[b_idx],
                                f_tracks
                            )

                            teams_initialized = True

                            print("Teams Initialized successfully.")

                            break

                # -----------------------
                # 5. TEAM + BALL LOGIC
                # -----------------------
                for i, player_dict in enumerate(tracks['players']):

                    if teams_initialized:

                        for tid, p_data in player_dict.items():

                            # override if needed
                            if tid == 94:
                                team_id = 1
                            else:
                                team_id = team_assigner.get_player_team(
                                    mini_batch[i],
                                    p_data['bbox'],
                                    tid
                                )

                            tracks['players'][i][tid]['team_id'] = team_id
                            tracks['players'][i][tid]['team_color'] = team_assigner.team_colors[team_id]

                    # Ball assignment
                    ball_bbox = tracks['ball'][i][1].get('bbox', [])

                    assigned_player = player_assigner.assign_ball_to_player(
                        player_dict,
                        ball_bbox
                    )

                    if assigned_player != -1:

                        tracks['players'][i][assigned_player]['has_ball'] = True

                        p_team = tracks['players'][i][assigned_player].get('team_id', 1)

                        overall_team_ball_control.append(p_team)

                    else:

                        overall_team_ball_control.append(
                            overall_team_ball_control[-1]
                            if overall_team_ball_control else 0
                        )

                # -----------------------
                # 6. DRAW OUTPUT
                # -----------------------
                annotated = tracker.draw_annotations(
                    mini_batch,
                    tracks,
                    overall_team_ball_control,
                    frame_count - 30
                )

                annotated = camera_movement_estimator.draw_camera_movement(
                    annotated,
                    cam_move
                )

                # ✅ DRAW SPEED & DISTANCE
                speed_and_distance_estimator.draw_speed_and_distance(
                    annotated,
                    tracks
                )

                for f in annotated:
                    writer.write(f)

                mini_batch = []

                gc.collect()

        # -----------------------
        # 7. PROCESS FINAL FRAMES
        # -----------------------
        if mini_batch:

            print(f"Processing final {len(mini_batch)} frames...")

            tracks = tracker.get_object_tracks(mini_batch)

            if camera_movement_estimator:

                cam_move = camera_movement_estimator.get_camera_movement(
                    mini_batch,
                    read_from_stub=False
                )

                tracker.add_position_to_tracks(tracks)

                view_transformer.add_transformed_position_to_tracks(tracks)

            tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

            # ✅ SPEED & DISTANCE FINAL BATCH
            speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

            for i, player_dict in enumerate(tracks['players']):

                if teams_initialized:

                    for tid, p_data in player_dict.items():

                        if tid == 94:
                            team_id = 1
                        else:
                            team_id = team_assigner.get_player_team(
                                mini_batch[i],
                                p_data['bbox'],
                                tid
                            )

                        tracks['players'][i][tid]['team_id'] = team_id
                        tracks['players'][i][tid]['team_color'] = team_assigner.team_colors[team_id]

                ball_bbox = tracks['ball'][i][1].get('bbox', [])

                assigned_player = player_assigner.assign_ball_to_player(
                    player_dict,
                    ball_bbox
                )

                if assigned_player != -1:

                    overall_team_ball_control.append(
                        tracks['players'][i][assigned_player].get('team_id', 1)
                    )

                else:

                    overall_team_ball_control.append(
                        overall_team_ball_control[-1]
                        if overall_team_ball_control else 0
                    )

            annotated = tracker.draw_annotations(
                mini_batch,
                tracks,
                overall_team_ball_control,
                frame_count - len(mini_batch)
            )

            if camera_movement_estimator:

                annotated = camera_movement_estimator.draw_camera_movement(
                    annotated,
                    cam_move
                )

            # ✅ DRAW SPEED & DISTANCE
            speed_and_distance_estimator.draw_speed_and_distance(
                annotated,
                tracks
            )

            for f in annotated:
                writer.write(f)

    except Exception as e:

        traceback.print_exc()

    finally:

        writer.release()

        print(f"--- Analysis Complete. Total Frames: {frame_count} ---")


if __name__ == "__main__":

    main()