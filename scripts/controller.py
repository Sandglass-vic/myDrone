#!/usr/bin/python
# -*- encoding: utf8 -*-

from scipy.spatial.transform import Rotation as R
from collections import deque
from enum import Enum
import rospy
import cv2
import numpy as np
import math
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from json import load


class ControllerNode:
    class FlightState(Enum):
        WAITING = 1
        NAVIGATING = 2
        # Fire pos
        DETECTING_TARGET = 3
        # Balls
        DETECTING_OBJECT = 4
        LANDING = 5

    def __init__(self):
        rospy.init_node('controller_node', anonymous=True)
        rospy.logwarn('Controller node set up.')

        # The pose of drone in the world's coordinate system
        self.R_wu_ = R.from_quat([0, 0, 0, 1])
        self.t_wu_ = np.zeros([3], dtype=np.float64)

        # For target detection
        self.bridge_ = CvBridge()
        self.image_ = None
        # Window positions
        self.window_x_list_ = (1.75, 4.25, 6.75)
        self.win_index_ = -1
        # Ball positions
        self.ball_pos_set_ = ((6.5, 7, 1.72), (), (5, 9.5, 1),
                              (4, 11, 1.72), (1, 14.5, 0.2))
        self.ball_index_ = -1
        # Answers
        self.detected_ball_num__ = 0
        self.ball_colors_ = ['n', 'n', 'n', 'n', 'n']
        # HSV range of balls
        self.red_color_range_ = ((0, 43, 46), (6, 255, 255))
        self.blue_color_range_ = ((100, 43, 46), (124, 255, 255))
        self.yellow_color_range_ = ((26, 43, 46), (34, 255, 255))
        # Camera
        self.cam_diff_ = 0
        self.cam_height_ = self.t_wu_[2] + self.cam_diff_

        # For navigation
        self.flight_state_ = self.FlightState.WAITING
        self.nav_nodes_ = None  # a list of 4-dimensional list[x,y,z,yaw]
        self.next_nav_node_ = None
        self.fixed_nav_routine_ = None
        self.readRoutineFile()
        self.next_state_ = None

        # Publications and subscriptions
        self.is_begin_ = False
        self.commandPub_ = rospy.Publisher(
            '/tello/cmd_string', String, queue_size=100)  # Make commands
        self.answerPub_ = rospy.Publisher(
            '/tello/target_result', String, queue_size=1)
        self.image_result_pub_ = rospy.Publisher(
            "/target_location/image_result", Image, queue_size=10)
        self.poseSub_ = rospy.Subscriber(
            '/tello/states', PoseStamped, self.poseCallback)  # Receive pose info
        self.imageSub_ = rospy.Subscriber(
            '/iris/usb_cam/image_raw', Image, self.imageCallback)  # Receive the image captured by camera
        self.startcmdSub_ = rospy.Subscriber(
            '/tello/cmd_start', Bool, self.startcommandCallback)  # Receive start command

        # Main loop
        rate = rospy.Rate(0.3)
        while not rospy.is_shutdown():
            if self.is_begin_:
                self.decide()
            rate.sleep()
        rospy.logwarn('Controller node shut down.')

    def readRoutineFile(self):
        self.fixed_nav_routine_ = load(open("./routine.json", "r"))
        for routine_set in self.fixed_nav_routine_.values():
            map(deque(), routine_set)

    # Updates
    def updateBallIndex(self):
        if self.ball_index_ == 0:
            self.ball_index_ += 2
        else:
            self.ball_index_ += 1

    def updateCamHeight(self):
        self.height_cam = self.t_wu_[2] + self.cam_diff_

    def adjustZ(self):
            # Adjust z coordinate
            while True:
                z_diff = self.next_nav_node_.nav_pos[2] - self.t_wu_[2]
                if abs(z_diff) < 0.3:
                    return True
                else:
                    commands = ['up ', 'down ']
                    command_index = 0 if z_diff > 0 else 1
                    self.publishCommand(
                        commands[command_index] + str(int(abs(100*z_diff))))
                    return False

    def adjustYaw(self, target_yaw):
        while True:
            yaw = self.R_wu_.as_euler('zyx', degrees=True)[0]
            yaw_diff = yaw - target_yaw if yaw - \
                target_yaw > 0 else yaw - target_yaw + 360
            if abs(yaw_diff) < 10:
                return True
            else:
                # clockwise and counterclockwise
                command_str = 'cw %d' if yaw_diff > 0 else 'ccw %d'
                self.publishCommand(command_str % (int(abs(yaw_diff))))
                return False

    def adjustXY(self):
        # Fly to our destination on the x-y plane
        while True:
            delta_x = self.next_nav_node_.nav_pos[0] - self.t_wu_[0]
            delta_y = self.next_nav_node_.nav_pos[1] - self.t_wu_[1]
            rho_diff = int((delta_x**2 + delta_y**2)**0.5)
            if abs(rho_diff) < 0.3:
                return True
            else:
                commands = ['forward ', 'back ']
                command_index = 0 if rho_diff > 0 else 1
                self.publishCommand(
                    commands[command_index] + str(int(abs(100*rho_diff))))
                return False

    def switchNavigatingState(self):
        if len(self.nav_nodes_) == 0:
            self.flight_state_ = self.next_state_
        else:
            self.next_nav_node_ = self.nav_nodes_.popleft()
            self.flight_state_ = self.FlightState.NAVIGATING

    # Detections
    def detectTarget(self):
        # Detect the red dot above the window
        if self.image_ is None:
            return False
        image_copy = self.image_.copy()
        height = image_copy.shape[0]
        width = image_copy.shape[1]

        frame = cv2.resize(image_copy, (width, height),
                           interpolation=cv2.INTER_CUBIC)  # 将图片缩放
        frame = cv2.GaussianBlur(frame, (3, 3), 0)  # 高斯模糊
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间
        h, s, v = cv2.split(frame)  # 分离出各个HSV通道
        v = cv2.equalizeHist(v)  # 直方图化
        frame = cv2.merge((h, s, v))  # 合并三个通道

        frame = cv2.inRange(frame, self.red_color_range_[
                            0], self.red_color_range_[1])  # 对原图像和掩模进行位运算
        opened = cv2.morphologyEx(
            frame, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算
        closed = cv2.morphologyEx(
            opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算
        (image, contours, hierarchy) = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找出轮廓

        # 在contours中找出最大轮廓
        contour_area_max = 0
        area_max_contour = None
        for c in contours:  # 遍历所有轮廓
            contour_area_temp = math.fabs(cv2.contourArea(c))  # 计算轮廓面积
            if contour_area_temp > contour_area_max:
                contour_area_max = contour_area_temp
                area_max_contour = c

        isTargetFound = False
        if area_max_contour is not None:
            if contour_area_max > 50:
                isTargetFound = True

        if isTargetFound:
            target = 'Red'
            ((centerX, centerY), rad) = cv2.minEnclosingCircle(
                area_max_contour)  # 获取最小外接圆
            cv2.circle(image_copy, (int(centerX), int(centerY)),
                       int(rad), (0, 255, 0), 2)  # 画出圆心
            win_dist = [abs(self.t_wu_[0]-win_x+0.5)
                            for win_x in self.window_x_list_]
            self.win_index_ = win_dist.index(min(win_dist))
            info_str = 'Target detected. Window index = %d' % self.win_index_
            rospy.logfatal(info_str)
        else:
            target = 'None'
            pass

        self.image_result_pub_.publish(self.bridge_.cv2_to_imgmsg(image_copy))
        return isTargetFound

    def color_area(self, color):
        # Return the max area of the given color in self.image
        if self.image_ is None:
            return
        image_copy = self.image_.copy()
        height = image_copy.shape[0]
        width = image_copy.shape[1]

        frame = cv2.resize(image_copy, (width, height),
                           interpolation=cv2.INTER_CUBIC)  # 将图片缩放
        frame = cv2.GaussianBlur(frame, (3, 3), 0)  # 高斯模糊
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间
        h, s, v = cv2.split(frame)  # 分离出各个HSV通道
        v = cv2.equalizeHist(v)  # 直方图化
        frame = cv2.merge((h, s, v))  # 合并三个通道

        if color == "r":
            frame = cv2.inRange(frame, self.red_color_range_[
                                0], self.red_color_range_[1])
        elif color == 'b':
            frame = cv2.inRange(frame, self.blue_color_range_[
                                0], self.blue_color_range_[1])
        else:
            frame = cv2.inRange(frame, self.yellow_color_range_[
                                0], self.yellow_color_range_[1])

        opened = cv2.morphologyEx(
            frame, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算
        closed = cv2.morphologyEx(
            opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算
        (image, contours, hierarchy) = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找出轮廓
        # 在contours中找出最大轮廓
        contour_area_max = 0
        area_max_contour = None
        for c in contours:  # 遍历所有轮廓
            contour_area_temp = math.fabs(cv2.contourArea(c))  # 计算轮廓面积
            if contour_area_temp > contour_area_max:
                contour_area_max = contour_area_temp
                area_max_contour = c
        return contour_area_max

    def detectObject(self):
        # Detect the ball of index self.ball_index and store the result
        red_area = self.color_area('r')
        blue_area = self.color_area('b')
        yellow_area = self.color_area('y')
        color = None
        if red_area < 10 and blue_area < 10 and yellow_area < 10:
            color = 'e'
        else:
            self.detected_ball_num__ += 1
            if red_area > blue_area and red_area > yellow_area:
                color = 'r'
            elif yellow_area > red_area and yellow_area > blue_area:
                color = 'y'
            else:
                color = 'b'
        self.ball_colors_[self.ball_index_] = color

    # Main function
    def decide(self):
        if self.flight_state_ == self.FlightState.WAITING:
            self.publishCommand('takeoff')
            self.win_index_ = 0
            self.nav_nodes_ = self.fixed_nav_routine_["detect_window"][self.win_index_]
            self.next_state_ = self.FlightState.DETECTING_TARGET
            self.switchNavigatingState()

        elif self.flight_state_ == self.FlightState.NAVIGATING:
            if not self.adjustZ():
                return
            # Calculate transition_yaw from the position now to the destination and adjust yaw
            transition_yaw = math.atan2(self.next_nav_node_.nav_pos[1] - self.t_wu_[1],
                                        self.next_nav_node_.nav_pos[0] - self.t_wu_[0]) / math.pi * 180
            if not self.adjustYaw(transition_yaw):
                return
            if not self.adjustXY():
                return
            # Set our drone's yaw the same as the yaw of our next_nav_node_
            if not self.adjustYaw(self.next_nav_node_.nav_yaw):
                return
            self.switchNavigatingState()

        elif self.flight_state_ == self.FlightState.DETECTING_TARGET:
            if self.detectTarget():
                # Navigate to pos A
                self.next_state_ = self.FlightState.DETECTING_OBJECT
                self.updateBallIndex()
                self.nav_nodes_ = self.fixed_nav_routine_["through_window"][self.win_index_]
                self.switchNavigatingState()
            else:
                if self.win_index_ >= 3:
                    rospy.loginfo('Detection failed, ready to land.')
                    self.flight_state_ = self.FlightState.LANDING
                else:  # Continue detecting
                    self.win_index_ += 1
                    self.nav_nodes_ = self.fixed_nav_routine_["detect_window"][self.win_index_]

        elif self.flight_state_ == self.FlightState.DETECTING_OBJECT:
            self.detectObject()
            if self.ball_index_ == 4:
                self.nav_nodes_ = self.fixed_nav_routine_["normal_land"][0]
                self.next_state_ = self.FlightState.LANDING
                self.switchNavigatingState()
                return
            if self.detected_ball_num__ == 3 and self.ball_index_ == 3:
                self.nav_nodes_ = self.fixed_nav_routine_["pre_land"][0]
                self.next_state_ = self.FlightState.LANDING
                self.switchNavigatingState()
                return
            self.updateBallIndex()
            self.next_state_ = self.FlightState.DETECTING_OBJECT
            self.nav_nodes_ = self.fixed_nav_routine_["detect_ball"][self.ball_index_]
            self.switchNavigatingState()

        elif self.flight_state_ == self.FlightState.LANDING:
            self.answerPub_.publish(''.join(self.ball_colors_))
            self.publishCommand('land')
        else:
            pass

    # For publications and subscriptions
    def publishCommand(self, command_str):
        msg = String()
        msg.data = command_str
        self.commandPub_.publish(msg)
        info_vector = ["WAITING", "NAVIGATING",
            "DETECTING_TARGET", "DETECTING_OBJECT", "LANDING"]
        rospy.logwarn("State: %s" % info_vector[self.flight_state_]
        rospy.loginfo("Command: %s" % command_str)
        rospy.loginfo(
            f"Drone pose: {self.t_wu_} yaw={self.R_wu_.as_euler('zyx', degrees=True)[0]}")

    def poseCallback(self, msg):
        self.t_wu_=np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.R_wu_=R.from_quat(
            [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        pass

    def imageCallback(self, msg):
        try:
            self.image_=self.bridge_.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as err:
            print(err)

    def startcommandCallback(self, msg):
        self.is_begin_=msg.data

if __name__ == '__main__':
    controller=ControllerNode()
