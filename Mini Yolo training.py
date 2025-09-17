import urllib.request
import zipfile
import os

# URL of COCO128 dataset
url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
filename = "coco128.zip"

# Download
if not os.path.exists(filename):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, filename)

# Extract
print("Extracting...")
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall(".")

print("Done! Dataset extracted to coco128/ folder.")

from ultralytics import YOLO

# Pre-trained model load
model = YOLO("yolov8n.pt")

# Train
model.train(
    data="coco128.yaml",
    epochs=50,
    imgsz=640,
    project="YOLO Training",
    name="exp1"
)
