import cv2
import numpy as np
import os
cap = cv2.VideoCapture('traffic.mp4')

car_cascade = cv2.CascadeClassifier('cars.xml')


s=0

while True:
    ret, im = cap.read()
   
    if (type(im) == type(None)):
        break
    image=cv2.resize(im,(600,300))

    img=cv2.resize(im,(600,300))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(gray, kernel, iterations=3)
    dilation = cv2.dilate(erosion, kernel, iterations=2)
    cars = car_cascade.detectMultiScale(dilation,1.1,3)

    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.putText(img, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 2, 255), 3)
        crop=image[y:y+h,x:x+w]
        crop=cv2.resize(crop,(600,300))
        cv2.imshow('image.jpg',crop)

        cv2.imwrite(os.path.join("C:/Users/Joel Joseph/Desktop/Temp/plates",'obj'+str(s)+'.jpg'),crop)
        s+=1
    cv2.imshow('video', img)
    
    
    if cv2.waitKey(33) == 27:
        break
cv2.destroyAllWindows()
cap.release()