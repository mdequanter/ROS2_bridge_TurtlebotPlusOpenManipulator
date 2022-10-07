import rospy

from sensor_msgs.msg import LaserScan

from geometry_msgs.msg import Twist

def next_step(msg):

    print("NEXT STEP HERE")

    rospy.sleep(10)

def move_forward_distance(distance):
    move.linear.x = distance
    pub.publish(move) 


def move_forward(msg):

    step2_front_scan = msg.ranges[0]

    print("Front scan: ", step2_front_scan)

    print("--------------------------------------------")

    rospy.loginfo("Moving towards wall!")

    while step2_front_scan > 0.3:

        print("Front_scan: ", step2_front_scan)

        move.linear.x = 0.05        

        pub.publish(move)

    move.linear.x = 0.0

    pub.publish(move)

    #rospy.loginfo("Wall is now within target!")

    #next_step(msg)

def callback(msg):

    move_forward_distance(1)


    front_scan = msg.ranges[0]

    min_scan = min(msg.ranges)

    a_front_scan = round(front_scan,1)

    a_min_scan = round(min_scan,1)

    print("Front scan: ", a_front_scan)

    print("Minimal scan: ", a_min_scan)

    print("--------------------------------------------")

    if (a_front_scan >= 1) :
        
        move.linear.x = 0.2
        move.angular.z = 0.0
        pub.publish(move)
    else :
        move.angular.z = 0.0
        move.linear.x =  0.0
        pub.publish(move)


move = Twist() 

laser_get = LaserScan()

rospy.init_node('obstacle_avoidance_node')

pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

sub = rospy.Subscriber("/scan", LaserScan, callback)

rospy.spin()