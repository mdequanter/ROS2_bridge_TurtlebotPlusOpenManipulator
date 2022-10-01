import cv2
import numpy as np
from djitellopy import tello
from time import sleep

frameWidth = 640
frameHeight = 480
#cap = cv2.VideoCapture(1)
#cap.set(3, frameWidth)
#cap.set(4, frameHeight)
#cap.set(10,150)

me = tello.Tello()
me.connect()
print(me.get_battery())

me.streamon()

'''
while True:
    img = me.get_frame_read().frame
    img = cv2.resize(img,(368,248))
    cv2.imshow("Image",img)
    cv2.waitKey(1)
'''

myColors = [[0,235,0,3,255,255]]

myColorValues = [[0,0,255]]

def findColor(img,myColors, myColorValues):
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []
    #for each of these colors we create a mask
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        x,y = getContours(mask)
        print ("find")
        cv2.circle(imgResult,(x,y),10,myColorValues[count],cv2.FILLED)
        if x!=0 and y!=0:
            newPoints.append([x,y,count])
        count += 1
        #cv2.imshow(str(color[0]),mask)


def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area>500:
           # cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x, y, w, h = cv2.boundingRect(approx)
    return x+w//2, y

def drawOnCanvas(myPoints,myColorValues):
    for point in myPoints:
        cv2.circle(imgResult, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED)

while True:
    img = me.get_frame_read().frame
    imgResult = img.copy()
    print("running")
    findColor(img,myColors,myColorValues)
    cv2.imshow("Result", imgResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break