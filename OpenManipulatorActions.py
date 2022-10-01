from math import exp
import os
import rclpy
import select
import sys
import threading
from time import sleep

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

present_joint_angle = [0.0, 0.0, 0.0, 0.0, 0.0]
goal_joint_angle = [0.0, 0.0, 0.0, 0.0, 0.0]
prev_goal_joint_angle = [0.0, 0.0, 0.0, 0.0, 0.0]
present_kinematics_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
goal_kinematics_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
prev_goal_kinematics_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

task_position_delta = 0.01  # meter
joint_angle_delta = 0.05  # radian
path_time = 0.8  # second


os.environ["ROS_DOMAIN_ID"] = "30"


e = """
Communications Failed
"""


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


def main():
    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    try:
        rclpy.init()
    except Exception as e:
        print(e)

    try:
        teleop_keyboard = TeleopKeyboard()
    except Exception as e:
        print(e)

    try:
        rclpy.spin_once(teleop_keyboard)

        n = len(sys.argv)
        print("Total arguments passed:", n)

        action = sys.argv[1] 

        print(sys.argv[1])
        
        
        

        
        if (action == "grip") :

            goal_joint_angle[0] = present_joint_angle[0]
            goal_joint_angle[1] = present_joint_angle[1]
            goal_joint_angle[2] = present_joint_angle[2]
            goal_joint_angle[3] = present_joint_angle[3]
            goal_joint_angle[4] = present_joint_angle[4]
            teleop_keyboard.send_tool_control_request()
            print ("gripper opened")
            sleep(2)

            goal_joint_angle[0] = -0.0
            goal_joint_angle[1] = 0.800
            goal_joint_angle[2] = 0.04
            goal_joint_angle[3] = -0.7

            pathtime = 2.0
            teleop_keyboard.send_goal_joint_space(pathtime)
            print ("STEP 1/3 done")
            sleep(2)          
            goal_joint_angle[4] = -0.003
            pathtime = 2.0
            teleop_keyboard.send_tool_control_request()
            print ("STEP 2/3 done")
            sleep(2)
            pathtime = 2.0
            goal_joint_angle[1] = 0
            goal_joint_angle[1] = -1.086
            goal_joint_angle[2] = 0.822
            goal_joint_angle[3] = 0.055223
            teleop_keyboard.send_goal_joint_space(pathtime)
            print ("STEP 3/3 done")
            teleop_keyboard.destroy_node()
            rclpy.shutdown()
            exit("SEND")

        if (action == "init") :

            pathtime = 2.0
            goal_joint_angle[0] = -0.2
            goal_joint_angle[1] = -1.175029552
            goal_joint_angle[2] = 1.259398
            goal_joint_angle[3] = 0.055223
            goal_joint_angle[4] = 0.010
            teleop_keyboard.send_goal_joint_space(pathtime)
            teleop_keyboard.send_tool_control_request()
            teleop_keyboard.destroy_node()
            rclpy.shutdown()            
            exit("SEND")

        if (action == "REST") :

            goal_joint_angle[0] = -0.011
            goal_joint_angle[1] =  0.144
            goal_joint_angle[2] = 0.288
            goal_joint_angle[3] =  1.138
            pathtime = 5.0
            teleop_keyboard.send_goal_joint_space(pathtime)
            goal_joint_angle[4] = 0.010
            teleop_keyboard.send_tool_control_request()
            teleop_keyboard.destroy_node()
            rclpy.shutdown()            
            exit("SEND")

        if (action == "RIGHT") :

            pathtime = 2.0
            goal_joint_angle[0] = -1.47
            goal_joint_angle[1] = -1.175029552
            goal_joint_angle[2] = 1.259398
            goal_joint_angle[3] = 0.25
            goal_joint_angle[4] = 0.010
            teleop_keyboard.send_goal_joint_space(pathtime)
            teleop_keyboard.send_tool_control_request()
            teleop_keyboard.destroy_node()
            rclpy.shutdown()
            exit("SEND")

        if (action == "POSITIONABOVE") :

            pathtime = 2.0
            goal_joint_angle[0] = present_joint_angle[0]
            goal_joint_angle[1] = 0.982
            goal_joint_angle[2] = 0.009
            goal_joint_angle[3] = -0.830
            teleop_keyboard.send_goal_joint_space(pathtime)
   
            teleop_keyboard.destroy_node()
            rclpy.shutdown()
            exit("SEND")

        if (action == "TAKEUP") :

            pathtime = 2.0
            goal_joint_angle[0] = present_joint_angle[0]
            goal_joint_angle[1] = 1.239
            goal_joint_angle[2] = -0.703
            goal_joint_angle[3] = -0.453
            teleop_keyboard.send_goal_joint_space(pathtime)
            sleep(2)
            goal_joint_angle[4] = -0.001
            teleop_keyboard.send_tool_control_request()
            sleep(2)
            goal_joint_angle[0] = 0.0
            goal_joint_angle[1] = -1.175029552
            goal_joint_angle[2] = 1.259398
            goal_joint_angle[3] = 0.055223
            teleop_keyboard.send_goal_joint_space(pathtime)
            teleop_keyboard.destroy_node()
        
            rclpy.shutdown()
            exit("SEND")



        if (action == "PLACEOBJECTRIGHT") :

            pathtime = 2.0
            goal_joint_angle[0] = -1.47
            goal_joint_angle[1] = 1.121
            goal_joint_angle[2] = -0.580
            goal_joint_angle[3] = -0.301
            teleop_keyboard.send_goal_joint_space(pathtime)
            sleep(5)
            goal_joint_angle[4] = 0.010
            teleop_keyboard.send_tool_control_request()
            goal_joint_angle[0] = -1.47
            goal_joint_angle[1] = 0.05
            goal_joint_angle[2] = -0.0
            goal_joint_angle[3] = -0.301
            teleop_keyboard.send_goal_joint_space(pathtime)
            sleep(1)
            pathtime = 2.0
            goal_joint_angle[0] = 0.0
            goal_joint_angle[1] = -1.175029552
            goal_joint_angle[2] = 1.259398
            goal_joint_angle[3] = 0.055223
            goal_joint_angle[4] = 0.010
            teleop_keyboard.send_goal_joint_space(pathtime)
            
            teleop_keyboard.destroy_node()
            rclpy.shutdown()
            exit("SEND")

                        
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()
