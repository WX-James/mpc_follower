#include "mpc.h"
#include <ros/ros.h>

int main( int argc, char * argv[] )
{ 
  ros::init(argc, argv, "mpc_node");
  ros::NodeHandle nh;
  ros::Duration(2.0).sleep();

  MPC mpc_tracker;

  mpc_tracker.init(nh);
  
  ROS_WARN("MPC init done");

  ros::spin();

  return 0;
}