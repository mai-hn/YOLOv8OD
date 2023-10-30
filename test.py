import cv2
from ultralytics import YOLO
import os

knownWidth = 100
focalLength = 1000

# Load the YOLOv8 model
model = YOLO('weight/best.pt')

# Open the video file
video_path = "./videos/test1.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, show=False, save_txt=True, save=True, save_conf=True, save_crop=True, tracker="bytetrack.yaml")

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        for box in results[0].boxes.xywh:
                box = box.to('cpu').numpy()
                # 计算距离
                distance = (knownWidth * focalLength) / box[2]
                # 绘制距离
                cv2.putText(annotated_frame, f'{distance:.2f}m', (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()