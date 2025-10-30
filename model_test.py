import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(
    "/Users/austinstanleyhinson/Desktop/converted_keras/keras_model.h5",
    "/Users/austinstanleyhinson/Desktop/converted_keras/labels.txt"
)

offset = 20
imgSize = 300
labels = ["Hello","Thank you","No","Yes","Please", "Okay", "I love you"]

while True:
    success, img = cap.read()
    if not success:
        break

    imgOutput = img.copy()
    hands, _ = detector.findHands(img)  
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        
        ih, iw = img.shape[:2]
        x1 = max(x - offset, 0)
        y1 = max(y - offset, 0)
        x2 = min(x + w + offset, iw)
        y2 = min(y + h + offset, ih)

       
        if x2 <= x1 or y2 <= y1:
            cv2.imshow("Image", imgOutput)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            continue

        imgCrop = img[y1:y2, x1:x2]
        if imgCrop.size == 0:
            cv2.imshow("Image", imgOutput)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            continue

        
        imgWhite = np.ones((imgSize, imgSize, 3), dtype=np.uint8) * 255

        hC, wC = imgCrop.shape[:2]
        if wC == 0 or hC == 0:
            cv2.imshow("Image", imgOutput)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            continue

        aspect = hC / wC

        if aspect > 1:
            k = imgSize / hC
            wCal = int(math.ceil(k * wC))
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            hR, wR = imgResize.shape[:2]
            wR = min(wR, imgSize)
            wGap = (imgSize - wR) // 2
            imgWhite[:, wGap:wGap + wR] = imgResize[:, :wR]
        else:
            k = imgSize / wC
            hCal = int(math.ceil(k * hC))
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hR, wR = imgResize.shape[:2]
            hR = min(hR, imgSize)
            hGap = (imgSize - hR) // 2
            imgWhite[hGap:hGap + hR, :] = imgResize[:hR, :]

        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        label = labels[index] if 0 <= index < len(labels) else "?"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1.2, 2)
        bx1 = max(x - offset, 0)
        by1 = max(y - offset - (th + 20), 0)
        bx2 = min(bx1 + tw + 20, iw)
        by2 = min(by1 + th + 20, ih)
        cv2.rectangle(imgOutput, (bx1, by1), (bx2, by2), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, label, (bx1 + 10, by2 - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)

        # Hand bbox
        cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Debug views
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):  # Esc or q to quit
        break

cap.release()
cv2.destroyAllWindows()
