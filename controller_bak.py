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

    def __init__(self):
        rospy.init_node('controller_node', anonymous=True)
        rospy.logwarn('Controller node set up.')

        # Loop rate
        self.decide_rate_ = 0.3

        # The pose of drone in the world's coordinate system
        self.R_wu_ = R.from_quat([0, 0, 0, 1])
        self.t_wu_ = np.zeros([3], dtype=np.float64)

        # For navigation
        self.pid_Kp_ = 0.35
        self.pid_Ki_ = 0.05
        self.pid_Kd_ = 0.05
        self.allowed_pos_diff_ = 30
        self.allowed_yaw_diff_ = 10
        self.max_pos_adjustment_ = 100
        self.max_yaw_adjustment_ = 15
        self.last_height_ = 1.5
        self.flight_state_ = self.FlightState.WAITING
        self.nav_nodes_ = None  # a deque of list [dimension, pos]
        self.next_nav_node_ = None
        self.fixed_nav_routine_ = None
        self.routine_command_map_ = {"x": 0, "y": 1, "z": 2, "yaw": 3}
        self.commands_matrix_ = [["right ", "left "], [
            "forward ", "back "], ["up ", "down "], ["cw ", "ccw "]]
        self.readRoutineFile()
        self.next_state_ = None

        # For detection
        self.bridge_ = CvBridge()
        self.image_ = None
        self.win_index_ = -1
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

        # Main loop
        rate = rospy.Rate(self.decide_rate_)
        time.sleep(15)
        while not rospy.is_shutdown():
            # if self.is_begin_:
            self.decide()
            rate.sleep()
        rospy.logwarn('Controller node shut down.')

    def readRoutineFile(self):
        self.fixed_nav_routine_ = load(
            open(getPackagePath('uav_sim') + "/scripts/routine.json", "r"))
        for key in self.fixed_nav_routine_:
            self.fixed_nav_routine_[key] = map(
                deque, self.fixed_nav_routine_[key])

    # Updates
    def updateBallIndex(self):
        if self.ball_index_ == 0:
            self.ball_index_ += 2
        else:
            self.ball_index_ += 1

    def updateCamHeight(self):
        self.height_cam = self.t_wu_[2] + self.cam_diff_

    def updatePosDiff(self):
        return 100 * (self.next_nav_node_[1] - self.t_wu_[self.routine_command_map_[self.next_nav_node_[0]]])

    """ def updateRhoDiff(self):
        delta_x = self.next_nav_node_[0] - self.t_wu_[0]
        delta_y = self.next_nav_node_[1] - self.t_wu_[1]
        if self.xy_forward_ and delta_x > 0:
            return 100*(delta_x**2 + delta_y**2)**0.5
        if self.xy_forward_ and delta_x < 0:
            return -1*100*(delta_x**2 + delta_y**2)**0.5
        if not self.xy_forward_ and delta_x < 0:
            return -1*100*(delta_x**2 + delta_y**2)**0.5
        if not self.xy_forward_ and delta_x > 0:
            return 100*(delta_x**2 + delta_y**2)**0.5

    def updateTransitionYaw(self):
        yaw_diff = self.R_wu_.as_euler('zyx', degrees=True)[0] - math.atan2(self.next_nav_node_[1] - self.t_wu_[1],
                                                                            self.next_nav_node_[0] - self.t_wu_[0]) / math.pi * 180
        if abs(yaw_diff) > 180:
            sig = -1 if yaw_diff > 0 else 1
            yaw_diff += sig * 360
        return yaw_diff """

    def update90YawDiff(self):
        yaw_diff = self.R_wu_.as_euler('zyx', degrees=True)[
            0] - 90
        if abs(yaw_diff) > 180:
            sig = -1 if yaw_diff > 0 else 1
            yaw_diff += sig * 360
        return yaw_diff

    def updateTargetYawDiff(self):
        yaw_diff = self.R_wu_.as_euler('zyx', degrees=True)[
            0] - self.next_nav_node_[1]
        if abs(yaw_diff) > 180:
            sig = -1 if yaw_diff > 0 else 1
            yaw_diff += sig * 360
        return yaw_diff

    # Pid adjustment
    def resetHeight(self):
        if self.next_nav_node_[0] == "z":
            self.last_height_ = self.next_nav_node_[1]
            return True
        info = "Reset height to {}".format(self.last_height_)
        rospy.loginfo(info)
        diff = self.t_wu_[2] - self.last_height_
        if abs(diff) < self.allowed_pos_diff_:
            return True
        adjustment = int(diff)
        commands = self.commands_matrix_[2]
        command_index = 0 if adjustment > 0 else 1
        self.publishCommand(
            commands[command_index] + str(min(self.max_pos_adjustment_, abs(adjustment))))
        return False

    def resetYaw(self):
        if self.next_nav_node_[0] == "yaw":
            return True
        yaw_diff = self.update90YawDiff()
        if abs(yaw_diff) < self.allowed_yaw_diff_:
            return True
        adjustment = int(yaw_diff)
        commands = self.commands_matrix_[3]
        command_index = 0 if adjustment > 0 else 1
        self.publishCommand(
            commands[command_index] + str(min(self.max_yaw_adjustment_,abs(adjustment))))
        return False

    def adjustYaw(self):
        yaw_diff = self.updateTargetYawDiff()
        if abs(yaw_diff) < self.allowed_yaw_diff_:
            return True
        adjustment = int(yaw_diff)
        commands = self.commands_matrix_[3]
        command_index = 0 if adjustment > 0 else 1
        self.publishCommand(
            commands[command_index] + str(min(self.max_yaw_adjustment_,abs(adjustment))))
        return False

    def adjustPos(self):
        self.resetHeight()
        key = self.next_nav_node_[
            0]
        allowed_diff = self.allowed_pos_diff_
        diff = self.updatePosDiff()
        commands = self.commands_matrix_[self.routine_command_map_[key]]
        if abs(diff) < allowed_diff:
            return True
        else:
            self.integral_ += diff
            delta_diff = self.last_diff_ - diff
            self.last_diff = diff
            adjustment = int(self.pid_Kp_*diff +
                             self.pid_Ki_*self.integral_+self.pid_Kd_*delta_diff)
            command_index = 0 if adjustment > 0 else 1
            self.publishCommand(
                commands[command_index] + str(min(self.max_pos_adjustment_,abs(adjustment))))
            return False

    def stop(self):
        time.sleep(0.2)
        self.publishCommand("stop")

    def switchNavigatingState(self):
        if self.nav_nodes_ == None or len(self.nav_nodes_) == 0:
            self.flight_state_ = self.next_state_
            info_vector = ["WAITING", "NAVIGATING",
                           "DETECTING_TARGET", "DETECTING_OBJECT", "LANDING"]
            rospy.logfatal(
                "State change->{}".format(info_vector[self.flight_state_.value]))
        else:
            self.next_nav_node_ = self.nav_nodes_.popleft()
            self.flight_state_ = self.FlightState.NAVIGATING
            self.resetYaw()
        # For PID use
        self.integral_ = 0
        self.last_diff_ = 0

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
            time.sleep(10)
            self.win_index_ = 0
            self.nav_nodes_ = self.fixed_nav_routine_[
                "detect_window"][self.win_index_]
            self.next_state_ = self.FlightState.DETECTING_TARGET
            self.switchNavigatingState()

        elif self.flight_state_ == self.FlightState.NAVIGATING:
            if self.next_nav_node_[0] == "yaw":
                if not self.adjustYaw():
                    return
            elif self.next_nav_node_[0] != "stop":
                if not self.adjustPos():
                    return          
            else:
                self.stop()
            self.switchNavigatingState()

        elif self.flight_state_ == self.FlightState.DETECTING_TARGET:
            if self.detectTarget():
                # Navigate to pos A
                self.next_state_ = self.FlightState.DETECTING_OBJECT
                self.updateBallIndex()
                self.nav_nodes_ = self.fixed_nav_routine_[
                    "through_window"][self.win_index_]
            else:
                if self.win_index_ >= 2:
                    rospy.loginfo('Detection failed, ready to land.')
                    self.nav_nodes_ = None
                    self.next_state_ = self.FlightState.LANDING
                else:  # Continue detecting
                    self.win_index_ += 1
                    self.nav_nodes_ = self.fixed_nav_routine_[
                        "detect_window"][self.win_index_]
                    self.next_state_ = self.FlightState.DETECTING_TARGET
            self.switchNavigatingState()

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
            self.nav_nodes_ = self.fixed_nav_routine_[
                "detect_ball"][self.ball_index_]
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
