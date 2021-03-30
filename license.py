import cv2
import imutils
import numpy as np
import pytesseract
import glob
import os
from unidecode import unidecode
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
path="C:/Users/Joel Joseph/Desktop/Temp/plates"
for i in(glob.glob(path + '**/*.*')):
    print(i)
    img = cv2.imread(i,cv2.IMREAD_COLOR)
    img = imutils.resize(img, width=1000)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray = cv2.bilateralFilter(gray, 13, 15, 15) 
    # thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #                 cv2.THRESH_BINARY,11,2)
    edged = cv2.Canny(gray, 50, 200) 
    cv2.imshow("edge.jpg",edged)
    contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None

    for c in contours:
            
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        print(approx)
        
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        detected = 0
        continue
    else:
        detected = 1
    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

    mask = np.zeros(gray.shape,np.uint8)
    new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    new_image = cv2.bitwise_and(img,img,mask=mask)
    cv2.imshow('image.jpg',new_image)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]
    Cropped=imutils.resize(Cropped,width=800)
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(Cropped,kernel,iterations = 1)
    cv2.imshow('dilate.jpg',dilation)
    erosion = cv2.erode(dilation,kernel,iterations = 2)
    cv2.imshow('erosion.jpg',erosion)
    closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('final',closing)

    text = pytesseract.image_to_string(closing,lang='eng', config='--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    print("Detected license plate Number is:",text)
    img = cv2.resize(img,(500,300))
    Cropped = cv2.resize(Cropped,(400,200))
    cv2.imshow('car',img)
    cv2.imshow('Cropped',Cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()