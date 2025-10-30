import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time 

capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
img_size = 300
counter = 0

folder = "/Users/austinstanleyhinson/Desktop/Sign_Language/Data/Hello"

while True:
    success, img = capture.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        ih, iw, _ = img.shape
        x1 = max(x - offset, 0)
        y1 = max(y - offset, 0)
        x2 = min(x + w + offset, iw)
        y2 = min(y + h + offset, ih)

        if x2 <= x1 or y2 <= y1:
            continue  

        imgCrop = img[y1:y2, x1:x2]
        imgWhite = np.ones((img_size, img_size, 3), np.uint8) * 255

        aspectratio = h/w

        if aspectratio > 1:
            k = img_size / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, img_size))
            hR, wR = imgResize.shape[:2]          
            wGap = math.ceil((img_size - wR) / 2)
            imgWhite[:, wGap:wR + wGap] = imgResize

        else:
            k = img_size / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (img_size, hCal))
            hR, wR = imgResize.shape[:2]          
            hGap = math.ceil((img_size - hR) / 2)
            imgWhite[hGap:hR + hGap, :] = imgResize


        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
    elif key == ord('q'):    
        break
    




