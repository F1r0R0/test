import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)

hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils

canvas = None

while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break

    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)

    # Пересоздаём полотно под текущее разрешение, если нужно
    if canvas is None or canvas.shape != image.shape:
        canvas = image.copy() * 0

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            h, w, _ = image.shape
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in handLms.landmark]

            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]

            # Расстояния между пальцами
            dist_draw = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
            dist_erase = math.hypot(index_tip[0] - middle_tip[0], index_tip[1] - middle_tip[1])

            if dist_draw < 40:
                cv2.circle(canvas, index_tip, 10, (0, 0, 255), -1)

            elif dist_erase < 40:
                cv2.circle(canvas, index_tip, 30, (0, 0, 0), -1)

            # Визуальные подсказки
            cv2.circle(image, thumb_tip, 8, (255, 0, 0), -1)
            cv2.circle(image, index_tip, 8, (0, 255, 0), -1)
            cv2.circle(image, middle_tip, 8, (0, 255, 255), -1)
            cv2.line(image, thumb_tip, index_tip, (255, 255, 255), 2)
            cv2.line(image, index_tip, middle_tip, (200, 200, 200), 1)

            draw.draw_landmarks(image, handLms, mp.solutions.hands.HAND_CONNECTIONS)

    combined = cv2.bitwise_or(image, canvas)
    cv2.imshow("Hand Draw", combined)

cap.release()
cv2.destroyAllWindows()
