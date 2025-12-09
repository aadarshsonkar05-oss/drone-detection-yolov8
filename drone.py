from ultralytics import YOLO
import cv2

def iou(a, b):
    # a,b = [x1,y1,x2,y2]
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_w = max(0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0, min(ya2, yb2) - max(ya1, yb1))
    inter = inter_w * inter_h
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

def nms_python(boxes, iou_thresh=0.45):
    # boxes: list of [x1,y1,x2,y2,conf]
    if not boxes:
        return []
    # sort by conf desc
    boxes = sorted(boxes, key=lambda x: x[:4], reverse=True)# reverse = true means high to low sort ho 
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        new_boxes =[]
        for b in boxes:
            overlap =iou(best[:4],b[:4])
            if overlap <iou_thresh:
                new_boxes.append(b)
        boxes = new_boxes
    return keep

# --------- run model ----------
model = YOLO("yolov8n.pt")

results = model.predict(
    source="sample.jpg",
    save=False,      # we will save our own annotated image not in the folder
    imgsz=1280,      # putted it bigger for small drones as they are small objects
    conf=0.18,       # lower to catch small objects (but still it increases noise)
    iou=0.45,
    augment=False,
)

# collect boxes
# ---------- COLLECT RAW BOXES 
raw_boxes = []
for r in results:
    for b in r.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        conf = float(b.conf[0])

        # NEW STRICT CONFIDENCE FILTER
        if conf < 0.70:     # ignore anything below 70%
            continue

        # optional safety: avoid tiny boxes
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            continue

        raw_boxes.append([x1, y1, x2, y2, conf])

print("Raw boxes:", len(raw_boxes))

clean = nms_python(raw_boxes, iou_thresh=0.45)
print("Boxes after NMS:", len(clean))

# draw final boxes (single clean box(s))
img = cv2.imread("sample.jpg")
for (x1,y1,x2,y2,conf) in clean:
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(img, f"{conf:.2f}", (x1, max(0, y1-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

cv2.imwrite("Final annotated image.jpg", img)
print("Saved Final annotated image.jpg")