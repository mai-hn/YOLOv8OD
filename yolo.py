import cv2
from ultralytics import YOLO
import numpy as np

def yolov8(video_stream, model, camera_matrix, dist_coeffs):

    # Load the YOLOv8 model
    model = YOLO(model)

    # Loop through the video frames
    for frame in video_stream:
        # Read a frame from the video
        nparr = np.frombuffer(frame, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)

        if frame is not None:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, show=False, save_txt=True, save=True, save_conf=True, save_crop=True, tracker="bytetrack.yaml")

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

if __name__ == '__main__':
    yolov8('./videos/test.mp4', './weight/best.pt')