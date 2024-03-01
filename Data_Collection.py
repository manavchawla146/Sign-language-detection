# Data Collection
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Video Capture
cap = cv2.VideoCapture(0) 
detector = HandDetector(maxHands=1) # Detect Only One Hand

offset = 20
imgSize = 300
path = "Data/C"
counter = 0

while True:
    # Read Video
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        # index of Detected Hand
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Crop Image Of Hand 
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        # White Cenvas  / Image
        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255

        imgCropShape = imgCrop.shape
 
        aspectRatio = h / w

        # Put Crop Image On White Cenvas For Train Model
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize. shape
            widthGape = math.ceil((imgSize - wCal)/2)
            imgWhite[:,widthGape:wCal+widthGape] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize. shape
            widthGape = math.ceil((imgSize - hCal)/2)
            imgWhite[widthGape:hCal+widthGape,:] = imgResize

        cv2.imshow("Crop Hand",imgCrop) # Crop Image
        cv2.imshow("White Img",imgWhite) # White Image

    cv2.imshow("image",img) 
    key = cv2.waitKey(1)
    
    # Collection Images 
    if key == ord("s"):
        counter += 1 
        cv2.imwrite(f'{path}/Image_{time.time()}.jpg',imgWhite) 
        print(counter)

        if counter == 300:
            print("Data Collected!")
            break
