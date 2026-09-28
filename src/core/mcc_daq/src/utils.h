#ifndef MCC_DAQ_UTILS_H
#define MCC_DAQ_UTILS_H

#include <unistd.h>
#include <rclcpp/rclcpp.hpp>

extern uint8_t G_INFO_VERBOSITY;

// ansi color codes
#define BLU "\033[0;34m"
#define GRN "\033[0;32m"
#define RED "\033[0;31m"
#define PUR "\033[0;35m"
// No Color
#define NC "\033[0m"

// rclcpp logging shims so the hardware code reads the same as before
#define ROS_INFO(...) RCLCPP_INFO(rclcpp::get_logger("mcc_daq"), __VA_ARGS__)
#define ROS_WARN(...) RCLCPP_WARN(rclcpp::get_logger("mcc_daq"), __VA_ARGS__)
#define ROS_ERROR(...) RCLCPP_ERROR(rclcpp::get_logger("mcc_daq"), __VA_ARGS__)
#define ROS_DEBUG(...) RCLCPP_DEBUG(rclcpp::get_logger("mcc_daq"), __VA_ARGS__)
#define ROS_INFO_STREAM(args) RCLCPP_INFO_STREAM(rclcpp::get_logger("mcc_daq"), args)

#define ROS_GREEN(mystr) ROS_INFO(GRN mystr NC)
#define ROS_INFO1(...) if (G_INFO_VERBOSITY >= 1) {ROS_INFO(__VA_ARGS__);}
#define ROS_INFO2(...) if (G_INFO_VERBOSITY >= 2) {ROS_INFO(__VA_ARGS__);}
#define ROS_INFO3(...) if (G_INFO_VERBOSITY >= 3) {ROS_INFO(__VA_ARGS__);}
#define ROS_WARN1(...) if (G_INFO_VERBOSITY >= 1) {ROS_WARN(__VA_ARGS__);}


void msleep(uint32_t milliseconds);

#endif //MCC_DAQ_UTILS_H
