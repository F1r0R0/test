import cv2

image = cv2.imread('images/face1.png')


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


results = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)

for (x, y, w, h) in results:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

cv2.imshow("Results", image)

cv2.waitKey(0)

