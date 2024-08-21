import cv2
from ultralytics import YOLO
from collections import defaultdict

# Load YOLO model
model = YOLO("yolov8m.pt")
cap = cv2.VideoCapture("kisisayma.MP4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

# Video writer
video_writer = cv2.VideoWriter("snail_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize counters
snail_counts = defaultdict(int)
counted_ids = set()


def point_in_polygon(x, y, polygon):
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.track(im0, persist=True, show=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            x, y, w, h = box
            if point_in_polygon(x, y, region_points) and track_id not in counted_ids:
                class_name = results[0].names[int(cls)]
                if class_name in ["round snail", "conical snail"]:
                    snail_counts[class_name] += 1
                    counted_ids.add(track_id)

    # Draw the region
    cv2.polylines(im0, [np.array(region_points, np.int32)], True, (0, 255, 0), 2)

    # Display counts
    cv2.putText(im0, f"Round Snails: {snail_counts['round snail']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                2)
    cv2.putText(im0, f"Conical Snails: {snail_counts['conical snail']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("Final counts:")
print(f"Round Snails: {snail_counts['round snail']}")
print(f"Conical Snails: {snail_counts['conical snail']}")