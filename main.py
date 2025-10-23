from ultralytics import YOLO
import cv2
import torch
import time


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используется устройство: {device}")

model = YOLO('yolo11l.pt')
model.to(device)

cap = cv2.VideoCapture('videos/videoplayback.webm')

colors = {
    "car": (0, 255, 0),
    "truck": (255, 0, 0),
    "bus": (0, 0, 255),
    "motorbike": (255, 255, 0),
    "person": (255, 0, 255)
}

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1920, 1080))

    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time

    results = model.track(
        frame,
        persist=True,
        conf=0.6,
        device=device,
        verbose=False
    )

    counts = {"car": 0, "truck": 0, "bus": 0, "motorbike": 0, "person": 0}

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            if label in counts:
                counts[label] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = colors.get(label, (0, 255, 0))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    frame, f"{label} ID:{track_id} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )
    x, y = 20, 40
    for label, count in counts.items():
        if count > 0:
            cv2.putText(
                frame,
                f"{label}: {count}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            y += 30

    text = f"FPS: {int(fps)}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(
        frame,
        text,
        (frame.shape[1] - tw - 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    cv2.imshow("Видео с ИИ", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
