from ultralytics import YOLO
from PIL import Image, ImageDraw
import csv
import os

image_path = "data/imagery/parking_lot.jpeg"
output_csv = "data/output/debug.csv"
output_debug = "data/output/debug_centers.png"

os.makedirs("data/output", exist_ok=True)

model = YOLO("yolov8m.pt")

results = model(
    image_path,
    save=True,
    imgsz=1280,
    conf=0.15,
    classes=[2]
)

result = results[0]
boxes = result.boxes

# Load original image for debugging
img = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(img)

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x_center", "y_center", "confidence"])

    for box in boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        writer.writerow([x_center, y_center, conf])

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # Draw center point
        r = 4
        draw.ellipse(
            [x_center - r, y_center - r, x_center + r, y_center + r],
            fill="blue",
            outline="blue"
        )

img.save(output_debug)

print(f"Saved CSV to: {output_csv}")
print(f"Saved debug image to: {output_debug}")
print(f"Detections saved: {len(boxes)}")
print(f"Original image size: {img.size}")