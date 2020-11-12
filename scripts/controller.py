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
from tools import getPackagePath
import time


class ControllerNode:
    class FlightState(Enum):
        WAITING = 0
        NAVIGATING = 1
        # Fire pos
        DETECTING_TARGET = 2
        # Balls
        DETECTING_OBJECT = 3
        LANDING = 4
        FINISHED = 5

    def __init__(self):
        rospy.init_node('controller_node', anonymous=True)
        rospy.logwarn('Controller node set up.')

        # Loop rate
        self.decide_rate_ = 2

        # The pose of drone in the world's coordinate system
        self.R_wu_ = R.from_quat([0, 0, 0, 1])
        self.t_wu_ = np.zeros([3], dtype=np.float64)

        # For navigation
        self.allowed_pos_error_ = 20
        self.allowed_reset_error_ = 40
        self.allowed_yaw_error_ = 10
        self.max_pos_adjustment_ = [30, 30, 50]
        self.max_yaw_adjustment_ = 180
        self.commands_= [["right ", "left "],["forward ", "back "],["up ","down "]]
        self.dimension_yaw_ = [None,None,None]
        self.turning_point_ = 125
        self.flight_state_ = self.FlightState.WAITING
        self.nav_nodes_ = None  # a deque of list [x/y/z/yaw, number]
        self.next_nav_node_ = None
        self.next_nav_pos_ = [1.7, 1.1, 1.55]
        self.fixed_nav_route_ = None
        self.readRoutineFile()
        self.next_state_ = None

        # For detection
        self.bridge_ = CvBridge()
        self.image_ = None
        self.win_index_ = -1
        self.ball_index_ = -1
        # Answers
        self.detected_ball_num__ = 0
        self.ball_colors_ = ['n', 'e', 'n', 'n', 'n']
        # HSV range of balls
        self.red_color_range_ = ((0, 43, 46), (6, 255, 255))
        self.blue_color_range_ = ((100, 43, 46), (124, 255, 255))
        self.yellow_color_range_ = ((26, 43, 46), (34, 255, 255))
        # Camera
        self.cam_diff_ = 0
        self.cam_height_ = self.t_wu_[2] + self.cam_diff_

        # Publications and subscriptions
        self.is_begin_ = False
        self.commandPub_ = rospy.Publisher(
            '/tello/cmd_string', String, queue_size=10)  # Make commands
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

        rate = rospy.Rate(self.decide_rate_)
        time.sleep(15)
        while not rospy.is_shutdown():
            self.decide()
            rate.sleep()
        # Main loop
        '''rate = rospy.Rate(self.decide_rate_)
        while not rospy.is_shutdown():
            if self.is_begin_:
                self.decide()
            rate.sleep()'''
        rospy.logwarn('Controller node shut down.')

    def readRoutineFile(self):
        self.fixed_nav_route_ = load(
            open(getPackagePath('uav_sim') + "/scripts/route.json", "r"))
        for key in self.fixed_nav_route_:
            self.fixed_nav_route_[key] = map(
                deque, self.fixed_nav_route_[key])

    # Updates
    def updateBallIndex(self):
        if self.ball_index_ == -1:
            self.ball_index_ = 2
        elif self.ball_index_ == 2:
            self.ball_index_ = 0
        elif self.ball_index_ == 0:
            self.ball_index_ = 3
        elif self.ball_index_ == 3:
            self.ball_index_ = 4
        else:
            assert(False)

    def updateCamHeight(self):
        self.height_cam = self.t_wu_[2] + self.cam_diff_

    def updatePositionError(self, dimension):
        assert(dimension >= 0 and dimension <= 2)
        return int(100 * (self.next_nav_node_[1] - self.t_wu_[dimension]))

    def updateYawError(self, target_yaw):
        yaw_error = self.R_wu_.as_euler('zyx', degrees=True)[
            0] - target_yaw
        if abs(yaw_error) > 180:
            sig = -1 if yaw_error > 0 else 1
            yaw_error += sig * 360
        return int(yaw_error)

    # Adjustment
    """ def adjustX(self):
        dimension = 0
        error = self.updatePositionError(dimension)
        if self.adjustZ():
            if abs(error) < self.allowed_pos_error_:
                return True
            else:
                if not self.adjustYaw(self.x_yaw_):
                    return False
                command_index = 0 if error > 0 else 1
                self.publishCommand(
                    self.x_commands[command_index] + str(min(self.max_pos_adjustment_[0], abs(error))))
                return False

    def adjustY(self):
        dimension = 1
        error = self.updatePositionError(dimension)
        if self.adjustZ():
            if abs(error) < self.allowed_pos_error_:
                return True
            else:
                if not self.adjustYaw(self.y_yaw_):
                    return False
                command_index = 0 if error > 0 else 1
                self.publishCommand(
                    self.y_commands[command_index] + str(min(self.max_pos_adjustment_[1], abs(error))))
                return False

    def adjustZ(self):
        dimension = 2
        error = self.updatePositionError(dimension)
        if abs(error) < self.allowed_pos_error_:
            return True
        else:
            commands = ["up ", "down "]
            command_index = 0 if error > 0 else 1
            self.publishCommand(
                commands[command_index] + str(min(self.max_pos_adjustment_[2], abs(error))))
            return False """

    def resetOthers(self, dimension):
        if dimension != 2:
            return True
        error = self.updatePositionError(dimension)
        if abs(error) > self.allowed_reset_error_:
            command_index = 0 if error > 0 else 1
            self.publishCommand(
                self.commands_[dimension][command_index] + str(abs(error)))
            return False
        return True

    def adjustPos(self, dimension):
        dimension_set = {0, 1, 2} # x, y, z
        dimension_set.remove(dimension)
        for d in dimension_set:
            if not self.resetOthers(d):
                return False
        error = self.updatePositionError(dimension)
        if abs(error) < self.allowed_pos_error_:
            return True
        else:
            command_index = 0 if error > 0 else 1
            self.publishCommand(
                self.commands_[dimension][command_index] + str(min(self.max_pos_adjustment_[dimension], abs(error))))
            return False

    def adjustYaw(self, target_yaw):
        yaw_error = self.updateYawError(target_yaw)
        if abs(yaw_error) < self.allowed_yaw_error_:
            return True
        commands = ["cw ", "ccw "]
        command_index = 0 if yaw_error > 0 else 1
        self.publishCommand(
            commands[command_index] + str(min(self.max_yaw_adjustment_, abs(yaw_error))))
        return False

    def switchNavigatingState(self):
        if self.nav_nodes_ == None or len(self.nav_nodes_) == 0:
            self.flight_state_ = self.next_state_
            info_vector = ["WAITING", "NAVIGATING",
                           "DETECTING_TARGET", "DETECTING_OBJECT", "LANDING", "FINISHED"]
            rospy.logfatal(
                "State change->{}".format(info_vector[self.flight_state_.value]))
        else:
            self.next_nav_node_ = self.nav_nodes_.popleft()
            self.flight_state_ = self.FlightState.NAVIGATING
            if len(self.next_nav_node_) == 1:
                self.publishCommand(self.next_nav_node_[0])
                self.switchNavigatingState()
            else:
                # X axis
                x_error = self.updatePositionError(0)
                if abs(x_error) > self.turning_point_:
                    if x_error > 0:
                        self.dimension_yaw_[0] = 0
                        self.commands_[0] = ["forward ", "back "]
                    else:
                        self.dimension_yaw_[0] = 180
                        self.commands_[0] = ["back ", "forward "]
                else:
                    self.dimension_yaw_[0] = 90
                    self.commands_[0] = ["right ", "left "]
                # Y axis
                y_error = self.updatePositionError(1)
                if y_error > 0:
                    self.dimension_yaw_[1] = 90
                    self.commands_[1] = ["forward ", "back "]
                else:
                    self.dimension_yaw_[1] = -90
                    self.commands_[1] = ["back ", "forward "]
                if self.next_nav_node_[0] != "yaw":
                    self.next_nav_pos_[
                    ord(self.next_nav_node_[0])-ord("x")] = self.next_nav_node_[1]

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
            info_str = 'Target detected. Window index = %d' % self.win_index_
            rospy.logfatal(info_str)
        else:
            target = 'None'
            info_str = 'Target didn\'t exist. Window index = %d' % self.win_index_
            rospy.logfatal(info_str)
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
        # Detect the ball of index self.ball_index_ and store the result
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
        info_str = 'Object detected. color = %s' % color
        rospy.logfatal(info_str)

    # Main function
    def decide(self):
        if self.flight_state_ == self.FlightState.WAITING:
            self.publishCommand('takeoff')
            time.sleep(10)
            self.win_index_ = 0
            self.nav_nodes_ = self.fixed_nav_route_[
                "detect_window"][self.win_index_]
            self.next_state_ = self.FlightState.DETECTING_TARGET
            self.switchNavigatingState()

        elif self.flight_state_ == self.FlightState.NAVIGATING:
            if self.next_nav_node_[0] == 'yaw':
                if not self.adjustYaw(self.next_nav_node_[1]):
                    return
            else:
                if not self.adjustPos(ord(self.next_nav_node_[0]) - ord("x")):
                    return
            self.switchNavigatingState()

        elif self.flight_state_ == self.FlightState.DETECTING_TARGET:
            if self.detectTarget() or self.win_index_ == 2:  # No need to detect
                # Navigate to pos A
                self.updateBallIndex()
                self.nav_nodes_ = self.fixed_nav_route_[
                    "through_window"][self.win_index_]
                self.next_state_ = self.FlightState.DETECTING_OBJECT
            else:
                # Continue detecting
                self.win_index_ += 1
                self.nav_nodes_ = self.fixed_nav_route_[
                    "detect_window"][self.win_index_]
                self.next_state_ = self.FlightState.DETECTING_TARGET
            self.switchNavigatingState()

        elif self.flight_state_ == self.FlightState.DETECTING_OBJECT:
            self.detectObject()
            if self.ball_index_ == 4:
                self.nav_nodes_ = self.fixed_nav_route_["normal_land"][0]
                self.next_state_ = self.FlightState.LANDING
                self.switchNavigatingState()
                return
            elif self.detected_ball_num__ == 3:
                self.nav_nodes_ = self.fixed_nav_route_["pre_land"][0]
                self.next_state_ = self.FlightState.LANDING
                self.switchNavigatingState()
                return
            self.updateBallIndex()
            self.next_state_ = self.FlightState.DETECTING_OBJECT
            self.nav_nodes_ = self.fixed_nav_route_[
                "detect_next_ball"][self.ball_index_]
            self.switchNavigatingState()

        elif self.flight_state_ == self.FlightState.LANDING:
            self.publishAnswer()
            self.publishCommand('land')
            self.nav_nodes_ = None
            self.next_state_ = self.FlightState.FINISHED
            self.switchNavigatingState()
        else:
            pass

    # For publications and subscriptions
    def publishCommand(self, command_str):
        msg = String()
        msg.data = command_str
        self.commandPub_.publish(msg)
        info_vector = ["WAITING", "NAVIGATING",
                       "DETECTING_TARGET", "DETECTING_OBJECT", "LANDING"]
        rospy.logwarn("State: {}".format(
            info_vector[self.flight_state_.value]))
        rospy.loginfo("Command: " + command_str)
        # Convert floats to ints
        pose = list(map(lambda f: round(f, 2), self.t_wu_))
        pose.append(int(self.R_wu_.as_euler('zyx', degrees=True)[0]))
        rospy.loginfo("Drone pose: {}".format(pose))
        rospy.loginfo("Next navigation node: {}".format(self.next_nav_node_))
        rospy.loginfo(
            "Win_index = {} Ball_index = {} Answer = {}".format(self.win_index_, self.ball_index_, ''.join(self.ball_colors_)))

    def publishAnswer(self):
        if self.detected_ball_num__ == 2:
            colors = {'r', 'b', 'y'}
            for color in self.ball_colors_:
                if color in colors:
                    colors.remove(color)
            self.ball_colors_[1] = colors.pop()
        ans = ''.join(self.ball_colors_)
        rospy.logfatal('Publish answer: ' + ans)
        self.answerPub_.publish(ans)

    def poseCallback(self, msg):
        self.t_wu_ = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.R_wu_ = R.from_quat(
            [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        pass

    def imageCallback(self, msg):
        try:
            self.image_ = self.bridge_.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as err:
            print(err)

    def startcommandCallback(self, msg):
        self.is_begin_ = msg.data


if __name__ == '__main__':
    controller = ControllerNode()
