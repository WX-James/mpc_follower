# mpc_follower

# 基于非线性模型预测控制的差速小车轨迹跟踪

## 1. 差速小车的运动学模型

​	对于两轮差速小车（四轮差速小车同样适用），其运动学模型如下：

<img src="/home/pi/Desktop/MPC_follower/model.jpg" style="zoom: 50%;" />

​	其中，差速小车的状态量有三个，即小车x方向位置、y方向位置和朝向Theta；小车的控制量有两个，即线速度V和角速度W，也是小车的 **cmd_vel**。

​	根据该离散时间模型以及当前的状态和控制，就可以递推的得到未来时刻的小车状态。

## 2. 代价函数

​	**MPC**的代价函数如下：

<img src="/home/pi/Desktop/MPC_follower/cost_function.png" style="zoom:50%;" />

​	其中，**X**k 矩阵表示小车k时刻的状态，，**U**k 矩阵表示小车k时刻的控制量；**Q **矩阵和 **R** 矩阵分别为状态和控制量的权重，也即我们更看重小车的跟踪误差，还是更看重小车的能耗。Lambda 是能量的附加权重项。

​	并且，这是一个累加形式的代价函数，对每一个预测时刻的状态和能量都需要求代价，得到预测步长N内所有的代价和。



## 3. MPC轨迹跟踪的Python代码

### （1）local_planner.py

​	该脚本根据规划器发送的期望状态( Traj_points )，以及小车的当前状态( Odometry )，调用 mpc.py 求解最优的控制量 cmd_vel。

#### 	1) 接收的Topic:

​	a. 小车的当前里程位置：/state_ukf/odom

​	该里程计Topic可由激光惯性里程计或者视觉惯性里程计得到，里程计的帧率越高越好，实测200帧下跟踪性能良好。

​	b. 期望的机器人状态：/mpc/traj_point

​	机器人状态也即我们期望小车跟踪的路点，该topic传递 Float32MultiArray类型的消息，也就是说我们给小车的期望状态是一个浮点数数组，具体形式为：[x1, y1, theta1, x2, y2, theta2, ....... , xN, yN, thetaN] , N是预测步长。

#### 	2) 发布的Topic:

​		a. 小车的控制输入：/cmd_vel

​		b. 模型预测控制器规划出来的局部路径：/mpc/local_path

### （2）mpc.py

​	该脚本根据 local_planner.py，得到当前状态以及期望状态，求解出最优控制量，返回给 local_planner.py。
