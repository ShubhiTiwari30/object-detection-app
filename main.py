from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Model load karo (pehle se download hoga)
model = YOLO("yolov8n.pt")

# Image load karo
img = cv2.imread("download.jpg")
if img is None:
    print("Image not found!")
    exit()

# YOLO detect
results = model(img)

# Detected image copy
detected_img = img.copy()

# Har detection ke liye
for r in results:
    for box in r.boxes:
        # Bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])  # class index
        label = model.names[cls]  # class name

        # Width, height, area
        w, h = x2 - x1, y2 - y1
        area = w * h

        # Crop region
        crop = img[y1:y2, x1:x2]

        # Average color (BGR)
        avg_color = np.mean(crop, axis=(0, 1))
        avg_color = tuple(map(int, avg_color))  # integer me convert

        print(f"Object: {label}, Size: {w}x{h}, Area: {area}, Avg Color (BGR): {avg_color}")

        # Draw bounding box
        cv2.rectangle(detected_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(detected_img, f"{label} {w}x{h}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Show result
plt.imshow(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

