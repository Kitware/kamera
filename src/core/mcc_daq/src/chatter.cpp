#include <signal.h>
#include <ros/ros.h>
#include <std_msgs/String.h>

int G_SIGINT_TRIGGERED = 0;

int main(int argc, char** argv)
{
    std::string node_ns = "/subsys0/chatter_in";
    std::string topic = "foobar";
    ros::init(argc, argv, "test");

    ros::NodeHandle nh("~");
    ros::Publisher p = nh.advertise<std_msgs::String> (topic, 1);
    ros::Publisher     bus_pub = nh.advertise<std_msgs::String> ("/bus", 10);


    signal(SIGINT, [](int i) {
        ROS_WARN("SIGINT triggered");
        G_SIGINT_TRIGGERED = 1;
    });
    while (ros::ok() && !G_SIGINT_TRIGGERED) {

        std::string inputString;
        std::cout << "Give input> ";
        std::getline(std::cin, inputString);
        std_msgs::String msg;

        if(!inputString.empty())
        {
            /** to test interfacing with the USB DAQ, use one of the following:
             * hi: set outpin high
             * lo: set outpin low
             * pu###: pulse outpin for ###ms
             * re: read analog on inpin
             */
            msg.data = inputString;
            p.publish(msg);
            bus_pub.publish(msg);
        }

        ros::spinOnce();
        if (G_SIGINT_TRIGGERED) {break;}
    }

    return 0;
}