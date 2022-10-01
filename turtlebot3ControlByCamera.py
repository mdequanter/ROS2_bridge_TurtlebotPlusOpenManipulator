import os
import select
import sys
import rclpy
import cv2
import numpy as np
from time import sleep

import subprocess
from rclpy.node import Node

from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile

if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

BURGER_MAX_LIN_VEL = 0.22
BURGER_MAX_ANG_VEL = 2.84

WAFFLE_MAX_LIN_VEL = 0.26
WAFFLE_MAX_ANG_VEL = 1.82

LIN_VEL_STEP_SIZE = 0.01
ANG_VEL_STEP_SIZE = 0.01

TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']

msg = """
Control Your TurtleBot3!
---------------------------
Moving around:
        w
   a    s    d
        x

w/x : increase/decrease linear velocity (Burger : ~ 0.22, Waffle and Waffle Pi : ~ 0.26)
a/d : increase/decrease angular velocity (Burger : ~ 2.84, Waffle and Waffle Pi : ~ 1.82)

space key, s : force stop

CTRL-C to quit
"""

e = """
Communications Failed
"""




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
            return x,y

    

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



cap = cv2.VideoCapture("http://192.168.1.100:8080/?action=stream")


lower_Blue = np.array([100, 100, 50])
upper_Blue = np.array([130, 255, 255])


lower_color = lower_Blue
upper_color = upper_Blue


myColorValues = [[0,0,255]]




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


def print_vels(target_linear_velocity, target_angular_velocity):
    print('currently:\tlinear velocity {0}\t angular velocity {1} '.format(
        target_linear_velocity,
        target_angular_velocity))


def make_simple_profile(output, input, slop):
    if input > output:
        output = min(input, output + slop)
    elif input < output:
        output = max(input, output - slop)
    else:
        output = input

    return output


def constrain(input_vel, low_bound, high_bound):
    if input_vel < low_bound:
        input_vel = low_bound
    elif input_vel > high_bound:
        input_vel = high_bound
    else:
        input_vel = input_vel

    return input_vel


def check_linear_limit_velocity(velocity):
    if TURTLEBOT3_MODEL == 'burger':
        return constrain(velocity, -BURGER_MAX_LIN_VEL, BURGER_MAX_LIN_VEL)
    else:
        return constrain(velocity, -WAFFLE_MAX_LIN_VEL, WAFFLE_MAX_LIN_VEL)


def check_angular_limit_velocity(velocity):
    if TURTLEBOT3_MODEL == 'burger':
        return constrain(velocity, -BURGER_MAX_ANG_VEL, BURGER_MAX_ANG_VEL)
    else:
        return constrain(velocity, -WAFFLE_MAX_ANG_VEL, WAFFLE_MAX_ANG_VEL)



settings = None
if os.name != 'nt':
    settings = termios.tcgetattr(sys.stdin)

rclpy.init()

qos = QoSProfile(depth=10)
node = rclpy.create_node('teleop_keyboard')
pub = node.create_publisher(Twist, 'cmd_vel', qos)

status = 0
target_linear_velocity = 0.0
target_angular_velocity = 0.0
control_linear_velocity = 0.0
control_angular_velocity = 0.0



frameCounter = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    frameResult = frame.copy()

    direction=""


    frameCounter = frameCounter+1
    x,y = findColor(frame,lower_color,upper_color)
    #print("x: "+str(x)+" y:"+str(y))
    if (frameCounter >=30) :
        frameCounter = 0
    

    cv2.imshow("Result", frameResult)



    key = get_key(settings)
    if key == 'w':
        target_linear_velocity = check_linear_limit_velocity(target_linear_velocity + LIN_VEL_STEP_SIZE)
        print("w detected")
        status = status + 1
        #print_vels(target_linear_velocity, target_angular_velocity)
    elif key == 'x':
        target_linear_velocity = check_linear_limit_velocity(target_linear_velocity - LIN_VEL_STEP_SIZE)
        print("x detected")
        status = status + 1
        #print_vels(target_linear_velocity, target_angular_velocity)
    elif direction == 'L':
        #target_angular_velocity =check_angular_limit_velocity(target_angular_velocity + ANG_VEL_STEP_SIZE)
        status = status + 1
        print ("L")
        #print_vels(target_linear_velocity, target_angular_velocity)
    elif direction == 'R':
        #target_angular_velocity = check_angular_limit_velocity(target_angular_velocity - ANG_VEL_STEP_SIZE)
        status = status + 1
        print ("R")
        #print_vels(target_linear_velocity, target_angular_velocity)
    elif direction == 'F':
        target_linear_velocity = 0.0
        control_linear_velocity = 0.0
        target_angular_velocity = 0.0
        control_angular_velocity = 0.0
        #print_vels(target_linear_velocity, target_angular_velocity)
    else:
        if (key == '\x03'):
            cap.release()
            cv2.destroyAllWindows()
            break

    if status == 20:
        print(msg)
        status = 0

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

    print ("publish Twist")
    pub.publish(twist)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

















def main():
    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    rclpy.init()

    qos = QoSProfile(depth=10)
    node = rclpy.create_node('teleop_keyboard')
    pub = node.create_publisher(Twist, 'cmd_vel', qos)

    status = 0
    target_linear_velocity = 0.0
    target_angular_velocity = 0.0
    control_linear_velocity = 0.0
    control_angular_velocity = 0.0

    center = 480

   

    try:
        print(msg)
        while(1):

            key = get_key(settings)
            if key == 'w':
                target_linear_velocity =\
                    check_linear_limit_velocity(target_linear_velocity + LIN_VEL_STEP_SIZE)
                status = status + 1
                print_vels(target_linear_velocity, target_angular_velocity)
            elif key == 'x':
                target_linear_velocity =\
                    check_linear_limit_velocity(target_linear_velocity - LIN_VEL_STEP_SIZE)
                status = status + 1
                print_vels(target_linear_velocity, target_angular_velocity)
            elif key == 'a':
                target_angular_velocity =\
                    check_angular_limit_velocity(target_angular_velocity + ANG_VEL_STEP_SIZE)
                status = status + 1
                print_vels(target_linear_velocity, target_angular_velocity)
            elif key == 'd':
                target_angular_velocity =\
                    check_angular_limit_velocity(target_angular_velocity - ANG_VEL_STEP_SIZE)
                status = status + 1
                print_vels(target_linear_velocity, target_angular_velocity)
            elif key == ' ' or key == 's':
                target_linear_velocity = 0.0
                control_linear_velocity = 0.0
                target_angular_velocity = 0.0
                control_angular_velocity = 0.0
                print_vels(target_linear_velocity, target_angular_velocity)
            else:
                if (key == '\x03'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break

            if status == 20:
                print(msg)
                status = 0

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

            pub.publish(twist)

    except Exception as e:
        print(e)

    finally:
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0

        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0

        pub.publish(twist)

        if os.name != 'nt':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


if __name__ == '__main__':
    main()
