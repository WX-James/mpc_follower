#!/usr/bin/env python
import rospy
from std_msgs.msg import Bool, Float64, Float32MultiArray
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Point, Twist
from nav_msgs.msg import Path, Odometry, OccupancyGrid
import numpy as np
import tf
from MPC import MPC
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker,MarkerArray
from std_srvs.srv import SetBool
import math

class Local_Planner():
    def __init__(self):
        self.replan_period = rospy.get_param('/local_planner/replan_period', 0.020)
        self.curr_state = np.zeros(5)
        self.z = 0.0
        self.N = 50
        self.goal_state = np.zeros([self.N,4])
        self.desired_global_path = [ np.zeros([300,4]) , 0]
        self.have_plan = False
        self.robot_state_set = False
        self.ref_path_set = False
        self.is_end=0

        self.__timer_replan = rospy.Timer(rospy.Duration(self.replan_period), self.__replan_cb)
        self.__pub_local_path = rospy.Publisher('mpc/local_path', Path, queue_size=10)
        self.__pub_rtc_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self._sub_odom = rospy.Subscriber('/state_ukf/odom', Odometry, self.__odom_cb)
        self._sub_traj_waypts = rospy.Subscriber('/mpc/traj_point', Float32MultiArray, self._vomp_path_callback)
        
        self.control_cmd = Twist()
        

    def __odom_cb(self,data):
        # 拿到机器人的里程位置
        self.robot_state_set = True
        self.curr_state[0] = data.pose.pose.position.x
        self.curr_state[1] = data.pose.pose.position.y
        # 四元数转rpy，拿到偏航角
        roll, pitch, self.curr_state[2] = self.quart_to_rpy(
            data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w)
        self.curr_state[3] = 0.0
        self.curr_state[4] = 0.0
        self.z = data.pose.pose.position.z


    def quart_to_rpy(self, x, y, z, w):
        r = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        p = math.asin(2*(w*y-z*x))
        y = math.atan2(2*(w*z+x*y), 1-2*(z*z+y*y))
        return r, p, y


    def __replan_cb(self, event):
        # 定时启用MPC轨迹跟踪
        if self.robot_state_set and self.ref_path_set:
            self.choose_goal_state()        ##  gobal planning
            start_time = rospy.Time.now()
            states_sol, input_sol = MPC(np.expand_dims(self.curr_state, axis=0),self.goal_state) ##  gobal planning
            end_time = rospy.Time.now()
            rospy.loginfo('[pHRI Planner] phri solved in {} sec'.format((end_time-start_time).to_sec()))

            if(self.is_end == 0):
                self.__publish_local_plan(input_sol,states_sol)
                self.cmd(input_sol)
            self.have_plan = True
        elif self.robot_state_set==False and self.ref_path_set==True:
            print("no pose")
        elif self.robot_state_set==True and self.ref_path_set==False:
            print("no path")
        else:
            print("no path and no pose")
        

    def __publish_local_plan(self,input_sol,state_sol):
        local_path = Path()
        sequ = 0
        local_path.header.stamp = rospy.Time.now()
        local_path.header.frame_id = "/world"

        for i in range(self.N):
            this_pose_stamped = PoseStamped()
            this_pose_stamped.pose.position.x = state_sol[i,0]
            this_pose_stamped.pose.position.y = state_sol[i,1]
            this_pose_stamped.pose.position.z = self.z+0.2 # 加0.2m的偏置，便于显示
            this_pose_stamped.header.seq = sequ
            sequ += 1
            this_pose_stamped.header.stamp = rospy.Time.now()
            this_pose_stamped.header.frame_id="/world"
            local_path.poses.append(this_pose_stamped)

        self.__pub_local_path.publish(local_path)

    def distance_global(self,c1,c2):
        distance = np.sqrt((c1[0]-c2[0])*(c1[0]-c2[0])+(c1[1]-c2[1])*(c1[1]-c2[1]))
        return distance
    

    def find_min_distance(self,c1):
        number =  np.argmin( np.array([self.distance_global(c1,self.desired_global_path[0][i]) for i in range(self.desired_global_path[1])]) )
        return number

    def choose_goal_state(self):
        # 选择轨迹中最近的点，并将其放在 N*1 的矩阵上
        # 说明：每次MPC都只跟踪轨迹中最近的一个点
        num = self.find_min_distance(self.curr_state)
        scale = 1
        num_list = []
        for i in range(self.N):  
            num_path = min(self.desired_global_path[1]-1,int(num+i*scale))
            num_list.append(num_path)
        if(num  >= self.desired_global_path[1]):
            self.is_end = 1
        for k in range(self.N):
            self.goal_state[k] = self.desired_global_path[0][num_list[k]]


    def _vomp_path_callback(self, data):
        if(len(data.data)!=0):
            self.ref_path_set = True
            size = len(data.data)//3
            self.desired_global_path[1]=size
            # 拿到规划出来的轨迹点
            car_yaw = self.curr_state[2]
            for i in range(size):
                self.desired_global_path[0][i,0]=data.data[3*(size-i)-3]
                self.desired_global_path[0][i,1]=data.data[3*(size-i)-2]
                
                if(data.data[3*(size-i)-1] - car_yaw > 3.14):
                    self.desired_global_path[0][i,2] = data.data[3*(size-i)-1] - 2.0*np.pi
                elif(data.data[3*(size-i)-1] - car_yaw < -3.14):
                    self.desired_global_path[0][i,2] = data.data[3*(size-i)-1] + 2.0*np.pi
                else:
                    self.desired_global_path[0][i,2]=data.data[3*(size-i)-1]
                
                self.desired_global_path[0][i,3]=0.0

    def cmd(self, data):
        self.control_cmd.linear.x = data[0][0]
        self.control_cmd.angular.z = data[0][1]
        self.__pub_rtc_cmd.publish(self.control_cmd)
# 

if __name__ == '__main__':
    rospy.init_node("MPC_Traj_follower")
    phri_planner = Local_Planner()

    rospy.spin()