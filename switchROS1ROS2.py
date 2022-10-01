import os

input("Switch to Openmanipulator, Press Enter ....")
feedback = os.system ("python /home/ubuntu/OneDrive/ROS2/OpenManipulatorILSF2022/OpenManipulatorActions.py init")
input("Switch to Turtlebot3, Press Enter ....")
feedback = os.system ("python /home/ubuntu/OneDrive/ROS2/OpenManipulatorILSF2022/turtlebot3ROS1_teleopKey.py")
input("Switch to Openmanipulator, Press Enter ....")
feedback = os.system ("python /home/ubuntu/OneDrive/ROS2/OpenManipulatorILSF2022/OpenManipulatorActions.py RIGHT")
input("Switch to Turtlebot3, Press Enter ....")
feedback = os.system ("python /home/ubuntu/OneDrive/ROS2/OpenManipulatorILSF2022/turtlebot3ROS1_teleopKey.py")
