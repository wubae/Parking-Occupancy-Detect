from ultralytics import YOLO
import csv
import os

model = YOLO("yolov8m.pt")

image_path = "data/imagery/parking_lot.jpeg"
output_csv = "data/output/detect.csv"

scale = 0.0001

os.makedirs("data/output", exist_ok=True)

results = model(
    image_path,
    save=True,
    imgsz=1280,
    conf=0.15,
    classes=[2]
)

boxes = results[0].boxes

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x_scaled", "y_scaled", "confidence"])

    for box in boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        x_scaled = x_center * scale
        
        image_height = 3753

        y_flipped = image_height - y_center
        y_scaled = y_flipped * scale

        writer.writerow([x_scaled, y_scaled, conf])

print("Scaled CSV ready")