#!/user/bin/env python
# coding: UTF-8

import sys
import numpy as np
import datetime
import time
import termios
from timeout_decorator import timeout, TimeoutError
import pandas as pd
import matplotlib.pyplot as plt

import rospy
import tf2_ros
import tf_conversions
import tf
from crazyswarm.msg  import GenericLogData

# 関連モジュールのインポート
from tools.Decorator import run_once
from frames_setup import Frames_setup
from tools.Mathfunction import LowPath_Filter, Mathfunction
from tools.Log import Log_data
from models import quadrotor_with_50cm_cable as model

# 定値制御
class Env_Experiment(Frames_setup):
    # このクラスの初期設定を行う関数
    def __init__(self, Texp, Tsam, num):
        # frames_setup, vel_controller の初期化
        super(Frames_setup, self).__init__()

        self.Tend = Texp
        self.Tsam = Tsam
        self.t = 0


        self.mathfunc = Mathfunction()
        
        # ! Initialization Lowpass Filter 
        self.LowpassP = LowPath_Filter()
        self.LowpassP.Init_LowPass2D(fc=5)
        self.LowpassV = LowPath_Filter()
        self.LowpassV.Init_LowPass2D(fc=5)
        self.LowpassE = LowPath_Filter()
        self.LowpassE.Init_LowPass2D(fc=5)
        self.LowpassL = LowPath_Filter()
        self.LowpassL.Init_LowPass2D(fc=5)
        self.LowpassVl = LowPath_Filter()
        self.LowpassVl.Init_LowPass2D(fc=5)
        self.Lowpassdq = LowPath_Filter()
        self.Lowpassdq.Init_LowPass2D(fc=5)


        self.set_frame()
        self.set_key_input()
        self.set_log_function()
        self.init_state()
        
        self.log = Log_data(num)
        
        time.sleep(0.5)

    # * set frame of crazyflie and paylad
    def set_frame(self):
        self.world_frame = Frames_setup().world_frame
        self.child_frame = Frames_setup().children_frame[0]
        self.payload_frame = Frames_setup().children_frame[1]
        self.tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(self.tfBuffer)
        time.sleep(0.5)

    # * keybord input function
    def set_key_input(self):
    
        self.fd = sys.stdin.fileno()

        self.old = termios.tcgetattr(self.fd)
        self.new = termios.tcgetattr(self.fd)

        self.new[3] &= ~termios.ICANON
        self.new[3] &= ~termios.ECHO

    def set_log_function(self):

        self.cmd_sub = rospy.Subscriber("/cf20/log1", GenericLogData, self.log_callback)

    def init_state(self):
        self.P = np.zeros(3)
        self.Ppre = np.zeros(3)
        self.Vrow = np.zeros(3)
        self.Vfiltered = np.zeros(3)
        self.R = np.zeros((3, 3))
        self.Euler = np.zeros(3)

        self.Pl = np.zeros(3)
        self.Plpre = np.zeros(3)
        self.Vl = np.zeros(3)
        self.Vrow_pre = np.zeros(3)
        self.Vl_filterd  = np.zeros(3)

        self.Height_drone = np.array([0.0, 0.0, 0.06])
        self.q = np.zeros(3)
        self.qpre = np.zeros(3)
        self.dqrow = np.zeros(3)
        self.dqrowpre = np.zeros(3)
        self.dq_filtered = np.zeros(3)

        try:
            quad = self.tfBuffer.lookup_transform(self.world_frame, self.child_frame, rospy.Time(0))
            load = self.tfBuffer.lookup_transform(self.world_frame, self.payload_frame, rospy.Time(0))

        # 取得できなかった場合は0.5秒間処理を停止し処理を再開する
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logerr('LookupTransform Error !')
            rospy.sleep(0.5)
            exit()

        self.P[0] = quad.transform.translation.x; self.P[1] = quad.transform.translation.y; self.P[2] = quad.transform.translation.z
        self.Quaternion = (quad.transform.rotation.x,quad.transform.rotation.y,quad.transform.rotation.z,quad.transform.rotation.w)
        self.Euler = tf_conversions.transformations.euler_from_quaternion(self.Quaternion)
        self.Eulerpre = self.Euler
        # self.R = tf_conversions.transformations.quaternion_matrix(self.Quaternion)[:3, :3]
        self.R = self.mathfunc.Euler2Rot(self.Euler)

        self.Pl[0] = load.transform.translation.x; self.Pl[1] = load.transform.translation.y; self.Pl[2] = load.transform.translation.z
        

# ------------------------------- ここまで　初期化関数 ---------------------

    def update_state(self):
        flag = False
        try:
            quad = self.tfBuffer.lookup_transform(self.world_frame, self.child_frame, rospy.Time(0))
            load = self.tfBuffer.lookup_transform(self.world_frame, self.payload_frame, rospy.Time(0))

        # 取得できなかった場合は0.5秒間処理を停止し処理を再開する
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logerr('LookupTransform Error !')
            flag = True
            

        # ! state of quadrotor 
        # position
        self.P[0] = quad.transform.translation.x; self.P[1] = quad.transform.translation.y; self.P[2] = quad.transform.translation.z
        self.P = self.LowpassP.LowPass2D(self.P, self.Tsam)
        # velocity
        self.Vrow = self.mathfunc.deriv(self.P, self.Ppre, self.dt)
        self.Vfiltered = self.LowpassV.LowPass2D(self.Vrow, self.Tsam)
        # attitude
        self.Quaternion = (quad.transform.rotation.x,quad.transform.rotation.y,quad.transform.rotation.z,quad.transform.rotation.w)
        self.Euler = self.LowpassE.LowPass2D(tf_conversions.transformations.euler_from_quaternion(self.Quaternion), self.Tsam)
        self.R = self.mathfunc.Euler2Rot(self.Euler)
        # previous states update
        self.Ppre[0] = self.P[0]; self.Ppre[1] = self.P[1]; self.Ppre[2] = self.P[2]
        self.Eulerpre = self.Euler

        # ! state of payload
        # position and velocity
        self.Pl[0] = load.transform.translation.x; self.Pl[1] = load.transform.translation.y; self.Pl[2] = load.transform.translation.z
        self.Pl = self.LowpassL.LowPass2D(self.Pl, self.Tsam)
        self.Vlrow = self.mathfunc.deriv(self.Pl, self.Plpre, self.dt)
        # self.Vlrow = self.mathfunc.Remove_outlier(self.mathfunc.deriv(self.Pl, self.Plpre, self.dt), self.Vrow_pre, 0.5)
        self.Vl_filterd = self.LowpassVl.LowPass2D(self.Vlrow, self.Tsam)

        # vector and vector velocity
        self.q = (self.Pl - (self.P-self.Height_drone))/np.linalg.norm((self.Pl - (self.P-self.Height_drone)))
        self.dqrow = self.mathfunc.deriv(self.q, self.qpre, self.dt)
        # self.dqrow = self.mathfunc.Remove_outlier(self.mathfunc.deriv(self.q, self.qpre, self.dt), self.dqrow_pre, 0.5)
        self.dq_filtered = self.Lowpassdq.LowPass2D(self.dqrow, self.Tsam)

        # previous state update
        self.Plpre[0] = self.Pl[0]; self.Plpre[1] = self.Pl[1]; self.Plpre[2] = self.Pl[2]
        self.qpre[0] = self.q[0];self.qpre[1] = self.q[1];self.qpre[2] = self.q[2]
        # self.dqrowpre[0] = self.dqrow[0]; self.dqrowpre[1] = self.dqrow[1]; self.dqrowpre[2] = self.dqrow[2]
        self.Vrow_pre = self.Vrow

        return flag

    def set_dt(self, dt):
        self.dt = dt

    def set_clock(self, t):
        self.t = t

    def log_callback(self, log):
        self.M = log.values

    def set_reference(self, controller,  
                            P=np.array([0.0, 0.0, 0.0]),   
                            V=np.array([0.0, 0.0, 0.0]), 
                            R=np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), 
                            Euler=np.array([0.0, 0.0, 0.0]), 
                            Wb=np.array([0.0, 0.0, 0.0]), 
                            Euler_rate=np.array([0.0, 0.0, 0.0]),
                            traj="circle",
                            controller_type="pid",
                            command = "hovering",
                            init_controller=True,
                            tmp_P=np.zeros(3)):
        if init_controller:
            controller.select_controller()
        if controller_type == "pid":
            if command =="hovering":
                controller.set_reference(P, V, R, Euler, Wb, Euler_rate, controller_type)    
            elif command == "land":
                controller.set_reference(P, V, R, Euler, Wb, Euler_rate, controller_type) 
            else:
                controller.set_reference(P, V, R, Euler, Wb, Euler_rate, controller_type)
        elif controller_type == "mellinger":
            controller.set_reference(traj, self.t, tmp_P)
        elif controller_type == "QCSL":
            controller.set_reference(traj, self.t, tmp_P)

        
    def take_log(self, ctrl):
        self.log.write_state(self.t, self.P, self.Vfiltered, self.R, self.Euler, np.zeros(3), np.zeros(3), self.M, self.Pl, self.Vl_filterd, self.q, self.dq_filtered)
        ctrl.log(self.log, self.t)
        
    def save_log(self):
      self.log.close_file()

    def get_input(self, input_thrust):
        flag = False
        try:
            input = self.input_with_timeout("key:")
            if input == "w":
                input_thrust += 0.1
            elif input == 'x':
                input_thrust -= 0.1
            elif input == "c":
                flag = True
                input_thrust = 0.0
            else:
                input = "Noinput"
        except TimeoutError:
            input = "Noinput"

        return input_thrust, flag

    def time_check(self, Tint, Tend):
        if Tint < self.Tsam:
            time.sleep(self.Tsam - Tint)
        if self.t > Tend:
            return True
        return False

    @run_once
    def land(self, controller, controller_type="pid"):
        controller.switch_controller("pid")
        self.set_reference(controller=controller, command="land", init_controller=True, P=self.land_P, controller_type="pid")
        
    @run_once
    def hovering(self, controller, P=np.array([0.0, 0.0, 1.0]), controller_type="mellinger"):
        self.set_reference(controller=controller, command="hovering", P=P, controller_type=controller_type)
        self.land_P = np.array([0.0, 0.0, model.L+0.5])

    def quad_takeoff(self, controller, controller_type="mellinger", Pinit=np.array([0.0, 0.0, 0.0])):
        controller.switch_controller(controller_type)
        self.set_reference(controller=controller, traj="takeoff", controller_type=controller_type, tmp_P=Pinit)
        self.land_P = np.array([0.0, 0.0, 0.1])

    def quad_takeoff_50cm(self, controller, Pinit=np.array([0.0, 0.0, 0.0])):
        self.set_reference(controller=controller, traj="takeoff_50cm", controller_type="mellinger", tmp_P=Pinit)
        self.land_P = np.array([0.0, 0.0, 0.1])

    def quad_land(self, controller, controller_type="mellinger"):
        controller.switch_controller(controller_type)
        self.set_reference(controller=controller, traj="land", controller_type=controller_type, init_controller=False, tmp_P=np.array([self.P[0], self.P[1], 0.0]))

    def quad_land_50cm(self, controller):
        self.set_reference(controller=controller, traj="land_50cm", controller_type="mellinger", init_controller=False, tmp_P=np.array([self.P[0], self.P[1], 0.0]))

    def quad_tack_circle(self, controller, controller_type="mellinger", flag=False):
        controller.switch_controller(controller_type)
        self.set_reference(controller=controller, traj="circle", controller_type=controller_type, init_controller=flag)

    def quad_tack_straight(self, controller, flag=False):
        self.set_reference(controller=controller, traj="straight", controller_type="mellinger", init_controller=flag)

    def quad_stop_track(self, controller, controller_type="mellinger"):
        controller.switch_controller(controller_type)
        self.set_reference(controller=controller, traj="stop", controller_type=controller_type, init_controller=False, tmp_P=np.array([self.P[0], self.P[1], self.P[2]]))
        self.land_P[0:2] = self.P[0:2]

    def payload_track_circle(self, controller, controller_type="QCSL", flag=False):
        controller.switch_controller(controller_type)
        self.set_reference(controller=controller, traj="circle", controller_type=controller_type, init_controller=flag)

    def payload_track_straight(self, controller, controller_type="QCSL", flag=False):
        controller.switch_controller(controller_type)
        self.set_reference(controller=controller, traj="straight", controller_type=controller_type, init_controller=flag)

    def payload_stop_track(self, controller, controller_type="QCSL"):
        controller.switch_controller(controller_type)
        self.set_reference(controller=controller, traj="stop", controller_type=controller_type, init_controller=False, tmp_P=np.array([self.Pl[0], self.Pl[1], self.Pl[2]]))
        self.land_P[0:2] = self.P[0:2]

    def paylaod_track_hover_payload(self, controller, controller_type="QCSL", flag=False):
        controller.switch_controller(controller_type)
        self.set_reference(controller=controller, traj="hover", controller_type=controller_type, init_controller=flag)

    def payload_takeoff(self, controller, controller_type="QCSL", Pinit=np.array([0.0, 0.0, 0.0])):
        controller.switch_controller(controller_type)
        self.set_reference(controller=controller, traj="takeoff", controller_type=controller_type, tmp_P=Pinit)
        self.land_P = np.array([0.0, 0.0, 0.1])
    
    def payload_land(self, controller, controller_type="QCSL"):
        controller.switch_controller(controller_type)
        self.set_reference(controller=controller, traj="land", controller_type=controller_type, init_controller=False, tmp_P=np.array([self.P[0], self.P[1], 0.0]))