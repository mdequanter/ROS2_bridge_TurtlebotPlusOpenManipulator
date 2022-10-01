from math import exp
import os
import rclpy
import select
import sys
import threading
from time import sleep
import cv2
import numpy as np

from open_manipulator_msgs.msg import KinematicsPose, OpenManipulatorState
from open_manipulator_msgs.srv import SetJointPosition, SetKinematicsPose
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState

if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty


marge = 10

present_joint_angle = [0.0, 0.0, 0.0, 0.0, 0.0]
goal_joint_angle = [0.0, 0.0, 0.0, 0.0, 0.0]
prev_goal_joint_angle = [0.0, 0.0, 0.0, 0.0, 0.0]
present_kinematics_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
goal_kinematics_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
prev_goal_kinematics_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

settings = None
camera =  "http://192.168.1.100:8080/?action=stream"

#camera =  "http://192.168.0.73:8080/?action=stream"
task_position_delta = 0.004  # meter
joint_angle_delta = 0.002  # radian
path_time = 0.1  # second


os.environ["ROS_DOMAIN_ID"] = "30"


cap = cv2.VideoCapture(camera)
cap.set(cv2.CAP_PROP_FPS, 1)

ret, frame = cap.read()

frameHeight = frame.shape[0]
frameWidth = frame.shape[1]

centerX = (frameWidth / 2) - 10 
centerY = frameHeight / 2


print('Frame Height       : ',frameHeight)
print('Frame Width        : ',frameWidth)


e = """
Communications Failed
"""


lower_Blue = np.array([77, 116, 185])
upper_Blue = np.array([114, 247, 255])

wasteBinRedLower = np.array([0, 97, 63])
wasteBinRedUpper = np.array([63, 255, 255])



lower_color = wasteBinRedLower
upper_color = wasteBinRedUpper


myColorValues = [[0,0,255]]



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
    x1 = int((frameWidth/2)-50)
    y1 = int((frameHeight/2)-50)
    x2 = int((frameWidth/2)+50)
    y2 = int((frameHeight/2)+50)
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)

    return img


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



class TeleopKeyboard(Node):

    qos = QoSProfile(depth=10)
    settings = None


    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    def __init__(self):
        super().__init__('teleop_keyboard')
        key_value = ''

        # Create joint_states subscriber
        self.joint_state_subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            self.qos)
        self.joint_state_subscription

        # Create kinematics_pose subscriber
        self.kinematics_pose_subscription = self.create_subscription(
            KinematicsPose,
            'kinematics_pose',
            self.kinematics_pose_callback,
            self.qos)
        self.kinematics_pose_subscription

        # Create manipulator state subscriber
        self.open_manipulator_state_subscription = self.create_subscription(
            OpenManipulatorState,
            'states',
            self.open_manipulator_state_callback,
            self.qos)
        self.open_manipulator_state_subscription

        # Create Service Clients
        self.goal_joint_space = self.create_client(SetJointPosition, 'goal_joint_space_path')
        self.goal_task_space = self.create_client(SetKinematicsPose, 'goal_task_space_path')
        self.tool_control = self.create_client(SetJointPosition, 'goal_tool_control')
        self.goal_joint_space_req = SetJointPosition.Request()
        self.goal_task_space_req = SetKinematicsPose.Request()
        self.tool_control_req = SetJointPosition.Request()

    def send_goal_task_space(self):
        self.goal_task_space_req.end_effector_name = 'gripper'
        self.goal_task_space_req.kinematics_pose.pose.position.x = goal_kinematics_pose[0]
        self.goal_task_space_req.kinematics_pose.pose.position.y = goal_kinematics_pose[1]
        self.goal_task_space_req.kinematics_pose.pose.position.z = goal_kinematics_pose[2]
        self.goal_task_space_req.kinematics_pose.pose.orientation.w = goal_kinematics_pose[3]
        self.goal_task_space_req.kinematics_pose.pose.orientation.x = goal_kinematics_pose[4]
        self.goal_task_space_req.kinematics_pose.pose.orientation.y = goal_kinematics_pose[5]
        self.goal_task_space_req.kinematics_pose.pose.orientation.z = goal_kinematics_pose[6]
        self.goal_task_space_req.path_time = path_time

        try:
            self.goal_task_space.call_async(self.goal_task_space_req)
        except Exception as e:
            self.get_logger().info('Sending Goal Kinematic Pose failed %r' % (e,))

    def send_goal_joint_space(self, path_time):
        self.goal_joint_space_req.joint_position.joint_name = ['joint1', 'joint2', 'joint3', 'joint4', 'gripper']
        self.goal_joint_space_req.joint_position.position = [goal_joint_angle[0], goal_joint_angle[1], goal_joint_angle[2], goal_joint_angle[3], goal_joint_angle[4]]
        self.goal_joint_space_req.path_time = path_time

        try:
            self.goal_joint_space.call_async(self.goal_joint_space_req)
        except Exception as e:
            self.get_logger().info('Sending Goal Joint failed %r' % (e,))

    def send_tool_control_request(self):
        self.tool_control_req.joint_position.joint_name = ['joint1', 'joint2', 'joint3', 'joint4', 'gripper']
        self.tool_control_req.joint_position.position = [goal_joint_angle[0], goal_joint_angle[1], goal_joint_angle[2], goal_joint_angle[3], goal_joint_angle[4]]
        self.tool_control_req.path_time = path_time

        try:
            self.tool_control_result = self.tool_control.call_async(self.tool_control_req)

        except Exception as e:
            self.get_logger().info('Tool control failed %r' % (e,))

    def kinematics_pose_callback(self, msg):
        present_kinematics_pose[0] = msg.pose.position.x
        present_kinematics_pose[1] = msg.pose.position.y
        present_kinematics_pose[2] = msg.pose.position.z
        present_kinematics_pose[3] = msg.pose.orientation.w
        present_kinematics_pose[4] = msg.pose.orientation.x
        present_kinematics_pose[5] = msg.pose.orientation.y
        present_kinematics_pose[6] = msg.pose.orientation.z

    def joint_state_callback(self, msg):
        present_joint_angle[0] = msg.position[0]
        present_joint_angle[1] = msg.position[1]
        present_joint_angle[2] = msg.position[2]
        present_joint_angle[3] = msg.position[3]
        present_joint_angle[4] = msg.position[4]

    def open_manipulator_state_callback(self, msg):
        if msg.open_manipulator_moving_state == 'STOPPED':
            for index in range(0, 7):
                goal_kinematics_pose[index] = present_kinematics_pose[index]
            for index in range(0, 5):
                goal_joint_angle[index] = present_joint_angle[index]

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
    #print_present_values()
    return key

def print_present_values():
    print(usage)
    print('Joint Angle(Rad): [{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}]'.format(
        present_joint_angle[0],
        present_joint_angle[1],
        present_joint_angle[2],
        present_joint_angle[3],
        present_joint_angle[4]))
    print('Kinematics Pose(Pose X, Y, Z | Orientation W, X, Y, Z): {:.3f}, {:.3f}, {:.3f} | {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
        present_kinematics_pose[0],
        present_kinematics_pose[1],
        present_kinematics_pose[2],
        present_kinematics_pose[3],
        present_kinematics_pose[4],
        present_kinematics_pose[5],
        present_kinematics_pose[6]))


rclpy.init()
teleop_keyboard = TeleopKeyboard()
rclpy.spin_once(teleop_keyboard)

if os.name != 'nt':
    settings = termios.tcgetattr(sys.stdin)

goal_joint_angle[0] = present_joint_angle[0]
goal_joint_angle[1] = present_joint_angle[1]
goal_joint_angle[2] = present_joint_angle[2]
goal_joint_angle[3] = present_joint_angle[3]
goal_joint_angle[4] = present_joint_angle[4]

prev_goal_joint_angle[0] = goal_joint_angle[0]
prev_goal_joint_angle[1] = goal_joint_angle[1]
prev_goal_joint_angle[2] = goal_joint_angle[2]
prev_goal_joint_angle[3] = goal_joint_angle[3]


pathtime = 0.5
teleop_keyboard.send_goal_joint_space(pathtime)

while(cap.isOpened()):


    ret, frame = cap.read()
    #frameResult = frame.copy()
    #direction = findColor(frame,lower_color,upper_color)

    x,y,radius =  findColor(frame,lower_color,upper_color)

    #if (x >= centerX-marge and x <= centerX+marge and y >= centerY-marge and y <= centerY+marge ) :
    if (x >= centerX-marge and x <= centerX+marge) :
        cv2.destroyAllWindows()
        exit("centered")
    
    if (radius > 0) :   
        print("x:"+ str(x) + " y:"+ str(y) + "radius:" + str(radius))
        cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 0), 5)

        if (x >= centerX-marge and x <= centerX+marge ) :
            objectCentered = True
        else :
            objectCentered = False

        if (x < centerX-marge):
            goal_joint_angle[0] = prev_goal_joint_angle[0] + joint_angle_delta
            teleop_keyboard.send_goal_joint_space(pathtime)
            prev_goal_joint_angle[0] = goal_joint_angle[0]
        if (x > centerX+marge):
            goal_joint_angle[0] = prev_goal_joint_angle[0] - joint_angle_delta
            teleop_keyboard.send_goal_joint_space(pathtime)
            prev_goal_joint_angle[0] = goal_joint_angle[0] 
        #overruled.  only X is needed. We know exact height of garbage bin
        '''
        if (objectCentered == True) :  
            if (y > centerY+marge):
                print ("Move up")
                if (goal_joint_angle[1]<= -0.975) :
                    goal_joint_angle[1] = prev_goal_joint_angle[1] + joint_angle_delta
                    teleop_keyboard.send_goal_joint_space(pathtime)
                    prev_goal_joint_angle[1] = goal_joint_angle[1]
                else :
                    print("Move jount down")
                    print (goal_joint_angle[3])
                    goal_joint_angle[3] = prev_goal_joint_angle[3] + joint_angle_delta
                    teleop_keyboard.send_goal_joint_space(pathtime)
                    prev_goal_joint_angle[3] = goal_joint_angle[3]

            if (y < centerY-marge):

                if (goal_joint_angle[3] >= 0.055223) :
                    print("Move jount up")
                    print (goal_joint_angle[3])
                    goal_joint_angle[3] = prev_goal_joint_angle[3] - joint_angle_delta
                    teleop_keyboard.send_goal_joint_space(pathtime)
                    prev_goal_joint_angle[3] = goal_joint_angle[3]
                else :                   
                    print ("move down")
                    goal_joint_angle[1] = prev_goal_joint_angle[1] - joint_angle_delta
                    teleop_keyboard.send_goal_joint_space(pathtime)
                    prev_goal_joint_angle[1] = goal_joint_angle[1]
            '''




    frame = setMarkers(frame)
    cv2.imshow("ResultAlign", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            

if os.name != 'nt':
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

teleop_keyboard.destroy_node()
rclpy.shutdown()