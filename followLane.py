#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from math import *
from geometry_msgs.msg import Twist


move = Twist() 
pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)




class image_converter:

    def __init__(self):
        self.image_pub = rospy.Publisher("image_topic_2",Image)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image",Image,self.callback)

        #self.lowerLane = np.array([22, 93, 0])
        #self.upperLane = np.array([45, 255, 255])
        #self.lowerLane = np.array([22, 93, 0])
        #self.upperLane = np.array([45, 255, 255])
        #Yellow
        self.lowerLane = np.array([9, 16, 154])   
        self.upperLane = np.array([42, 175, 255])

        self.laneMargin = 5


    def followLane(self,imgFrame) :

        frameHeight = imgFrame.shape[0]
        frameWidth = imgFrame.shape[1]

        # filter bottom area. So only image just in front of robot.
        imgFrame = imgFrame[frameHeight-200:frameHeight, 100:frameWidth-100]
        
        frameHeight = imgFrame.shape[0]
        frameWidth = imgFrame.shape[1]

        imgFrameCopy = imgFrame
        imgFrame = cv2.GaussianBlur(imgFrame, (11, 11), 0)
        imgHSV = cv2.cvtColor(imgFrame,cv2.COLOR_BGR2HSV)
        count = 0
        newPoints = []
        #for each of these colors we create a mask
        kernel = np.ones((9,9),np.uint8)
        imgFrameYellow = cv2.inRange(imgHSV, self.lowerLane, self.upperLane)
        cv2.imshow("Yellow" , imgFrameYellow)
        sensitivity = 15
        #lower_white = np.array([0,0,255-sensitivity])
        #upper_white = np.array([255,sensitivity,255])
        #lower_white = np.array([0,0,138])
        lower_white = np.array([0,0,138])
        #upper_white = np.array([172,111,255])
        upper_white = np.array([172,40,255])

        imgFrameWhite = cv2.inRange(imgHSV, lower_white, upper_white)
        cv2.imshow("White" , imgFrameWhite)

        contoursYellow,hierarchy = cv2.findContours(imgFrameYellow,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        topX,topY,bottomX,bottomY = 0,0,frameWidth,frameHeight
        if len(contoursYellow) > 0:
            for cnt in contoursYellow:
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

        if (bottomX > (frameWidth/2 + self.laneMargin)) :
                print("R")
                move.linear.x = 0.2
                move.angular.z = -0.2
                #pub.publish(move)
        elif (bottomX < (frameWidth/2 - self.laneMargin)) :
                print("L")
                move.linear.x = 0.2
                move.angular.z = 0.2
                #pub.publish(move)                
        if (bottomX >= (frameWidth/2 - self.laneMargin) and bottomX <= (frameWidth/2 + self.laneMargin)  ) :
                print("F")
                move.linear.x = 0.2
                move.angular.z = 0.0
                #pub.publish(move)                
        return imgFrameCopy


    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        cv_image = self.followLane(cv_image)


        cv2.imshow("Image window", cv_image)
        cv2.waitKey(1)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)