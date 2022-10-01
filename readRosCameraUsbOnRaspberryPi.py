from turtle import xcor
import cv2
import numpy as np
from time import sleep
import subprocess
import rclpy
from rclpy.node import Node


#bashCommand = "export ROS_DOMAIN_ID=31"
#process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
#output, error = process.communicate()

center = 480



cap = cv2.VideoCapture("http://192.168.0.77:8080/?action=stream")


lower_Blue = np.array([100, 100, 50])
upper_Blue = np.array([130, 255, 255])


lower_color = lower_Blue
upper_color = upper_Blue


myColorValues = [[0,0,255]]

#ROS2 methods

#output = subprocess.check_output(['bash','-c', 'export ROS_DOMAIN_ID=31'])
rclpy.init()



def get_topic_list():
    node_dummy = Node("_ros2cli_dummy_to_show_topic_list")
    topic_list = node_dummy.get_topic_names_and_types()
    node_dummy.destroy_node()
    return topic_list


# Cam methods

def findColor(img,lower_color,upper_color):
    img = cv2.GaussianBlur(frame, (11, 11), 0)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []
    #for each of these colors we create a mask
    kernel = np.ones((9,9),np.uint8)
    mask = cv2.inRange(imgHSV, lower_color, upper_color)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    if len(cnts) > 0:
        # find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # only proceed if the radius meets a minimum size. Correct this value for your obect's size
        if radius > 0.5:
            cv2.circle(img, (int(x), int(y)), int(radius),(255,0,0), 2)
            if (x<400):
                print ("go left")
            elif (x>500):
                print ("go right")
            elif (x>400 and  x<500):
                print ("go forward")
    
    #cv2.imshow("Result", img)



def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0
    if len(cnts) > 0:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            print(area)
            if area>80:
                peri = cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,0.02*peri,True)
                x, y, w, h = cv2.boundingRect(approx)
        return x+w//2, y    




while(cap.isOpened()):
    ret, frame = cap.read()
    frameResult = frame.copy()
    findColor(frame,lower_color,upper_color)
    #cv2.imshow("Result", frameResult)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

rclpy.shutdown()




