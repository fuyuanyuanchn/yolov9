import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Load YOLO model
model = YOLO("yolov8m.pt")
cap = cv2.VideoCapture("kisisayma.MP4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("snail_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize counters
snail_counts = defaultdict(int)
counted_ids = set()

frame_count = 0

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    frame_count += 1
    print(f"Processing frame {frame_count}")

    results = model.track(im0, persist=True, show=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()

        print(f"Detected {len(boxes)} objects in frame {frame_count}")

        for box, track_id, cls in zip(boxes, track_ids, clss):
            if track_id not in counted_ids:
                class_name = results[0].names[int(cls)]
                if class_name in ["round snail", "conical snail"]:
                    snail_counts[class_name] += 1
                    counted_ids.add(track_id)

            # Draw bounding box
            x, y, w, h = box
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im0, f"{class_name} {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            print(f"Drew box for {class_name} {track_id} at ({x1}, {y1}, {x2}, {y2})")

    # Display counts
    cv2.putText(im0, f"Round Snails: {snail_counts['round snail']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                2)
    cv2.putText(im0, f"Conical Snails: {snail_counts['conical snail']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    # Display the image
    cv2.imshow("Snail Counting", im0)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("Final counts:")
print(f"Round Snails: {snail_counts['round snail']}")
print(f"Conical Snails: {snail_counts['conical snail']}")