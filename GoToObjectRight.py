import os
from turtle import xcor
import cv2
import numpy as np
from time import sleep
import subprocess
import rclpy
import sys
from rclpy.node import Node
from math import *

from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile

if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

#os.system ("python /home/ubuntu/OneDrive/ROS2/OpenManipulatorILSF2022/OpenManipulatorGripperAction.py init")

feedback = os.system ("python /home/ubuntu/OneDrive/ROS2/OpenManipulatorILSF2022/OpenManipulatorActions.py init")
sleep(3)

#os.environ["ROS_DOMAIN_ID"] = "31"


BURGER_MAX_LIN_VEL = 0.22
BURGER_MAX_ANG_VEL = 2.84

WAFFLE_MAX_LIN_VEL = 0.26
WAFFLE_MAX_ANG_VEL = 1.82

LIN_VEL_STEP_SIZE = 0.15
ANG_VEL_STEP_SIZE = 0.2

TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']
ROS_DOMAIN_ID = os.environ['ROS_DOMAIN_ID']


#if (ROS_DOMAIN_ID != "31") :
#    sys.exit("Wrong Robot selected")

center = 480
yClose = 580

gripperStarted = False
alignStarted = False
finalApproachStarted = False

step = 0

#camera =  "http://192.168.0.73:8080/?action=stream"
camera =  "http://192.168.1.100:8080/?action=stream"

cap = cv2.VideoCapture(camera)
cap.set(cv2.CAP_PROP_FPS, 1)


ret, frame = cap.read()

frameHeight = frame.shape[0]
frameWidth = frame.shape[1]

centerX = frameWidth / 2
centerY = frameHeight / 2


lower_Blue = np.array([77, 116, 185])
upper_Blue = np.array([114, 247, 255])

wasteBinRedLower = np.array([0, 97, 63])
wasteBinRedUpper = np.array([63, 255, 255])



lower_color = wasteBinRedLower
upper_color = wasteBinRedUpper


myColorValues = [[0,0,255]]




print ("Working on ROS DOMAIN : " + ROS_DOMAIN_ID)
print ("Turtlebot : " + TURTLEBOT3_MODEL)
print ("stream : " + camera)
print ("##############################")
sleep(3)



#ROS2 methods

def get_topic_list():
    node_dummy = Node("_ros2cli_dummy_to_show_topic_list")
    topic_list = node_dummy.get_topic_names_and_types()
    node_dummy.destroy_node()
    return topic_list

def check_angular_limit_velocity(velocity):
    if TURTLEBOT3_MODEL == 'burger':
        return constrain(velocity, -BURGER_MAX_ANG_VEL, BURGER_MAX_ANG_VEL)
    else:
        return constrain(velocity, -WAFFLE_MAX_ANG_VEL, WAFFLE_MAX_ANG_VEL)

def check_linear_limit_velocity(velocity):
    if TURTLEBOT3_MODEL == 'burger':
        return constrain(velocity, -BURGER_MAX_LIN_VEL, BURGER_MAX_LIN_VEL)
    else:
        return constrain(velocity, -WAFFLE_MAX_LIN_VEL, WAFFLE_MAX_LIN_VEL)

def constrain(input_vel, low_bound, high_bound):
    if input_vel < low_bound:
        input_vel = low_bound
    elif input_vel > high_bound:
        input_vel = high_bound
    else:
        input_vel = input_vel

    return input_vel

def make_simple_profile(output, input, slop):
    if input > output:
        output = min(input, output + slop)
    elif input < output:
        output = max(input, output - slop)
    else:
        output = input

    return output



def doMove(direction):

    target_linear_velocity = 0.0
    target_angular_velocity = 0.0
    control_linear_velocity = 0.0
    control_angular_velocity = 0.0


    if (direction == 'L') :
        target_angular_velocity = check_angular_limit_velocity(target_angular_velocity + ANG_VEL_STEP_SIZE)

    if (direction == 'R') :
        target_angular_velocity = check_angular_limit_velocity(target_angular_velocity - ANG_VEL_STEP_SIZE)

    if (direction == 'F') :
        target_linear_velocity = check_linear_limit_velocity(target_linear_velocity + LIN_VEL_STEP_SIZE)

    if (direction == 'B') :
        target_linear_velocity = check_linear_limit_velocity(target_linear_velocity - LIN_VEL_STEP_SIZE)

    if (direction == 'SLOWFORWARD') :
        target_linear_velocity = check_linear_limit_velocity(target_linear_velocity + (LIN_VEL_STEP_SIZE/2))



    twist = Twist()
    control_linear_velocity = make_simple_profile(
        control_linear_velocity,
        target_linear_velocity,
        (LIN_VEL_STEP_SIZE / 2.0))

    twist.linear.x = control_linear_velocity
    twist.linear.y = 0.0
    twist.linear.z = 0.0

    control_angular_velocity = make_simple_profile(
        control_angular_velocity,
        target_angular_velocity,
        (ANG_VEL_STEP_SIZE / 2.0))

    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = control_angular_velocity

    #print("angular: " + str(target_angular_velocity) + "linear:" + str(target_linear_velocity))

    pub.publish(twist)




# Cam methods

def setMarkers(img):

        
    # draw coordinates on screen to make it easy to navigate
    text = "x: 0, y:0"
    cv2.putText(img, text, (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
    text = "x: " + str(frameWidth) +", y:0"
    cv2.putText(img, text, (frameWidth-120, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    text = "x: 0, y:" + str(frameHeight)
    cv2.putText(img, text, (5, frameHeight-10),
        cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
    text = "x: " + str(frameWidth) +", y:" + str(frameHeight)
    cv2.putText(img, text, (frameWidth-120, frameHeight-10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # draw a box in center of 100x100px
    x1 = int(frameWidth-200)
    y1 = 0
    x2 = int(frameWidth-200)
    y2 = frameHeight
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

    return img

def findColor(img,lower_color,upper_color):
    radius,x,y = 0,0,0
    img = cv2.GaussianBlur(frame, (11, 11), 0)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []
    #for each of these colors we create a mask
    kernel = np.ones((9,9),np.uint8)
    mask = cv2.inRange(imgHSV, lower_color, upper_color)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
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
        #center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # only proceed if the radius meets a minimum size. Correct this value for your obect's size
        #print (radius)
        #print("x:"+ str(x) + "radius:" + str(radius))
        return x,y,radius
    else :
        return x,y,radius




def followLane(imgFrame,lower_color,upper_color) :

    imgFrameCopy = imgFrame
    imgFrame = cv2.GaussianBlur(imgFrame, (11, 11), 0)
    imgHSV = cv2.cvtColor(imgFrame,cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []
    #for each of these colors we create a mask
    kernel = np.ones((9,9),np.uint8)
    imgFrame = cv2.inRange(imgHSV, lower_color, upper_color)
    contours,hierarchy = cv2.findContours(imgFrame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    topX,topY,bottomX,bottomY = 0,0,frameWidth,frameHeight
    if len(contours) > 0:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if (area > 30) :
                if len(approx) == 4 :
                    (x, y, w, h) = cv2.boundingRect(approx)
                    x = int(x+(w/2))
                    y = int(y+(h/2))
                    if (x>0 and y>0):
                        if (y > topY) :
                            topY = y
                            topX = x
                        if (y < bottomY) :
                            bottomY = y
                            bottomX = x

        if (bottomX>0 and bottomY>0 and topY > 100) :
            theta=atan2((bottomY-frameHeight),(bottomX-frameWidth/2))
            angle = theta*57.29+90
            cv2.line(frame,(topX,topY), (bottomX,bottomY), (0, 255, 0), 9)
            #print ("theta: " + str(angle) + " topY: " + str(topY))
            if (angle < -30):
                doMove("L")
            elif (angle > -20):
                doMove("R")
            else :
                doMove("SLOWFORWARD")
        else :
            doMove("S")
    else :
        doMove("S")

    return imgFrameCopy


settings = None
if os.name != 'nt':
    settings = termios.tcgetattr(sys.stdin)
rclpy.init()

qos = QoSProfile(depth=10)
node = rclpy.create_node('teleop_keyboard')
pub = node.create_publisher(Twist, 'cmd_vel', qos)



lastDirection = "X"

doMove('F')


objectFound = False


while(cap.isOpened()):


    ret, frame = cap.read()
    #frameResult = frame.copy()
    #direction = findColor(frame,lower_color,upper_color)
    
    x,y,radius =  findColor(frame,lower_color,upper_color)

    


    if (radius > 0):
        #print("x:"+ str(x) + " y:"+ str(y) + "radius:" + str(radius))
        #print(" y:"+ str(y) + "radius:" + str(radius))
        cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 5)
        # Move to object until it is big, so very close
        if (step == 0) :
            if (radius > 5 and y > 180):
                objectFound = True
                doMove('S')
                feedback = os.system ("python /home/ubuntu/OneDrive/ROS2/OpenManipulatorILSF2022/OpenManipulatorActions.py RIGHT")
                #os.environ["ROS_DOMAIN_ID"] = "31"
                sleep(1)
                step = 1
                

    
    if (objectFound == True and step == 1) :
        doMove("SLOWFORWARD")
        if (radius >50) :
                if (x > frameWidth/2-100 and x < frameWidth/2+100):
                    doMove("S")
                    cv2.destroyAllWindows()
                    feedback = os.system ("python /home/ubuntu/OneDrive/ROS2/OpenManipulatorILSF2022/OpenManipulatorFollowObject.py")
                    sleep(5)
                    cv2.destroyAllWindows()
                    feedback = os.system ("python /home/ubuntu/OneDrive/ROS2/OpenManipulatorILSF2022/OpenManipulatorActions.py POSITIONABOVE")
                    sleep(3)
                    feedback = os.system ("python /home/ubuntu/OneDrive/ROS2/OpenManipulatorILSF2022/OpenManipulatorFollowObject.py")
                    sleep(3)
                    cv2.destroyAllWindows()
                    step = 2
    if (objectFound == True and step == 2) :
        feedback = os.system ("python /home/ubuntu/OneDrive/ROS2/OpenManipulatorILSF2022/OpenManipulatorActions.py TAKEUP")
        sleep(3)
        n = 0
        while n<35 :
            doMove("B")
            sleep(0.5)
            n=n+1 
        doMove("S")
        feedback = os.system ("python /home/ubuntu/OneDrive/ROS2/OpenManipulatorILSF2022/OpenManipulatorActions.py PLACEOBJECTRIGHT")
        sleep(3)
        feedback = os.system ("python /home/ubuntu/OneDrive/ROS2/OpenManipulatorILSF2022/OpenManipulatorActions.py REST")
        sleep(3)        
        exit()

    if (objectFound == False) :
        linesFrame = followLane(frame,lower_Blue, upper_Blue)

     

    #frame = setMarkers(frame)
    #cv2.imshow("Result", frame)


    cv2.imshow("Result", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

rclpy.shutdown()
