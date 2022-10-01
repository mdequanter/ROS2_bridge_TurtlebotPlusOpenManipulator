import os
from turtle import xcor
import cv2
import numpy as np
from time import sleep
import subprocess
import rclpy
import sys
from rclpy.node import Node

from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile

if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty


os.environ["ROS_DOMAIN_ID"] = "31"


BURGER_MAX_LIN_VEL = 0.22
BURGER_MAX_ANG_VEL = 2.84

WAFFLE_MAX_LIN_VEL = 0.26
WAFFLE_MAX_ANG_VEL = 1.82

LIN_VEL_STEP_SIZE = 0.2
ANG_VEL_STEP_SIZE = 0.2

TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']
ROS_DOMAIN_ID = os.environ['ROS_DOMAIN_ID']


camera =  "http://192.168.1.100:8080/?action=stream"


print ("Working on ROS DOMAIN : " + ROS_DOMAIN_ID)
print ("Turtlebot : " + TURTLEBOT3_MODEL)
print ("stream : " + camera)
print ("##############################")
sleep(3)

if (ROS_DOMAIN_ID != "31") :
    sys.exit("Wrong Robot selected")


os.system ("python /home/ubuntu/OneDrive/ROS2/OpenManipulatorILSF2022/OpenManipulatorGripperAction.py init")

center = 480
yClose = 580

gripperStarted = False

cap = cv2.VideoCapture(camera)

cap.set(cv2.CAP_PROP_FPS, 1)


lower_Blue = np.array([100, 100, 50])
upper_Blue = np.array([130, 255, 255])

wasteBinRedLower = np.array([0, 107, 124])
wasteBinRedUpper = np.array([30, 255, 255])



lower_color = lower_Blue
upper_color = upper_Blue


myColorValues = [[0,0,255]]

#ROS2 methods

def get_key(settings):
    if os.name == 'nt':
        return msvcrt.getch().decode('utf-8')
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

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

    print("angular: " + str(target_angular_velocity) + "linear:" + str(target_linear_velocity))

    pub.publish(twist)



    
    #target_angular_velocity = 0.0

    
    '''
    twist = Twist()
    control_linear_velocity = make_simple_profile(
        control_linear_velocity,
        target_linear_velocity,
        (LIN_VEL_STEP_SIZE / 2.0))

    twist.linear.z = 0.0

    control_angular_velocity = make_simple_profile(
        control_angular_velocity,
        target_angular_velocity,
        (ANG_VEL_STEP_SIZE / 2.0))

    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = control_angular_velocity

    pub.publish(twist)
    '''
        




# Cam methods

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



settings = None
if os.name != 'nt':
    settings = termios.tcgetattr(sys.stdin)
rclpy.init()

qos = QoSProfile(depth=10)
node = rclpy.create_node('teleop_keyboard')
pub = node.create_publisher(Twist, 'cmd_vel', qos)



lastDirection = "X"

doMove('F')


while(cap.isOpened()):



    ret, frame = cap.read()
    #frameResult = frame.copy()
    #direction = findColor(frame,lower_color,upper_color)

    x,y,radius =  findColor(frame,lower_color,upper_color)

    if (radius > 0):

        #print("x:"+ str(x) + " y:"+ str(y) + "radius:" + str(radius))
        cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 5)

        if (gripperStarted == False) :
            if (y > yClose):
                doMove('S')
            elif(y < yClose and y and x > center-20 and x < center+20) :
                doMove('F')
            elif(y < yClose and x > center-20) :
                doMove('R')
            elif(y < yClose and x < center+20) :
                doMove('L')
            
            if (y < yClose+30 and y > yClose-30 and x > center-20 and x < center+20):
                doMove('S')
                print ("start gripper")
                os.environ["ROS_DOMAIN_ID"] = "30"
                print (os.environ['ROS_DOMAIN_ID'])
                os.system ("python /home/ubuntu/OneDrive/ROS2/OpenManipulatorILSF2022/OpenManipulatorGripperAction.py grip")
                gripperStarted = True
 
    cv2.imshow("Result", frame)

    #cv2.imshow("Result", frameResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

rclpy.shutdown()




