import cv2

def read_video(video_path):
    """
    A generator that reads a video frame by frame.
    This prevents OutOfMemory errors by only keeping ONE frame in RAM at a time.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame  # Yields the frame to the caller, then waits for next request
    
    cap.release()

def save_video(output_video_frames, output_video_path):
    """
    Note: In a streaming setup, we usually write frames as we go.
    This function is kept for compatibility if you have small clips.
    """
    if not output_video_frames:
        return
        
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, 
                          (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()