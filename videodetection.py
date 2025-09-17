from ultralytics import YOLO
import cv2
import numpy as np

# 1️⃣ Load YOLOv8 model
model = YOLO("yolov8n.pt")  # nano model (fast, lightweight)

# 2️⃣ Open webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Webcam open nahi ho paayi!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3️⃣ YOLO detection on the frame
    results = model(frame)

    # 4️⃣ Draw bounding boxes & info
    annotated_frame = frame.copy()
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Width, height, area
            w, h = x2 - x1, y2 - y1
            area = w * h

            # Crop region for average color
            crop = frame[y1:y2, x1:x2]
            avg_color = tuple(map(int, np.mean(crop, axis=(0, 1))))

            # Draw bounding box + label
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{label} {w}x{h}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 5️⃣ Show the frame
    cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6️⃣ Release resources
cap.release()
cv2.destroyAllWindows()


from ultralytics import YOLO
import cv2
import numpy as np

# 1️⃣ Load YOLOv8 model
model = YOLO("yolov8n.pt")

# 2️⃣ Video file path
video_path = "24333-341474163_tiny.mp4"  # yaha apna video ka naam likho
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video open nahi ho paayi!")
    exit()

# 3️⃣ Frame by frame process
while True:
    ret, frame = cap.read()
    if not ret:   # agar video khatam ho gaya
        break

    results = model(frame)
    annotated_frame = frame.copy()

    for r in results:
        for box in r.boxes:
            # bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # size
            w, h = x2 - x1, y2 - y1
            area = w * h

            # average color
            crop = frame[y1:y2, x1:x2]
            avg_color = tuple(map(int, np.mean(crop, axis=(0, 1))))

            print(f"Object: {label}, Size: {w}x{h}, Area: {area}, Avg Color: {avg_color}")

            # draw box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 4️⃣ Show video
    cv2.imshow("YOLOv8 Video Detection", annotated_frame)

    # press q to quit before video ends
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5️⃣ Release
cap.release()
cv2.destroyAllWindows()
