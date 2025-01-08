#ifndef MCC_DAQ_DAQ_NODE_H
#define MCC_DAQ_DAQ_NODE_H

#include <memory>
#include <mutex>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>

#include <roskv/envoy.h>
#include <custom_msgs/SetTriggerRate.h>
#include <custom_msgs/GSOF_EVT.h>

#include "usbdaq.h"

using std::shared_ptr;

bool is_simple_frac(double d);

class TriggerTimer {
private:
    ros::NodeHandlePtr node;
    UsbDaq             usbDaq;
    ros::Publisher     bus_pub;
    ros::ServiceServer srvTrigger;
    bool               trigger_is_running = false;
    const double       default_freq       = 0.5;
    const double       quick_idle_freq    = 10;
    double             freq_set           = default_freq;
    ros::Timer         timer;
    ros::Time          last_call          = ros::Time::now();
    ros::Time          last_edge          = ros::Time::now();
    ros::Time          next_edge          = ros::Time::now();
    ros::Duration      last_duration      = ros::Duration(0.5);
    int                flippy             = 0;
public:

    TriggerTimer(ros::NodeHandlePtr node_, UsbDaq &usbDaq1);

    bool is_running();

    bool call();

    void set_trigger_run(bool state);

    void set_trigger_run(const std_msgs::Bool::ConstPtr &msg);

    void set_trigger_freq(double frequency);

    void set_trigger_freq(const std_msgs::Float64::ConstPtr &msg);
    ros::Rate get_trigger_freq();
    ros::Duration get_trigger_dur();

    void set_trigger_period(double t_seconds);

    void set_trigger_period(ros::Duration duration);

    void trigger_tic(const ros::TimerEvent &event);

    bool setTriggerRate(custom_msgs::SetTriggerRate::Request &req,
                        custom_msgs::SetTriggerRate::Response &resp);

    void nop(const ros::TimerEvent &event);
    ros::Time get_next_edge();
    void next();
    void sleep();
    void spin_then_sleep();
    void sleep_until_edge(double granularity);
    void sleep_next();


    ros::Rate rate = ros::Rate(quick_idle_freq);
};

void cb_print(const ros::TimerEvent &e) {
    ROS_INFO("default callback");
}

class OneShotManager {
public:
    OneShotManager() = default;

    void erase(boost::uuids::uuid i);

    boost::uuids::uuid addOneShot(ros::NodeHandlePtr nhp,
                                    const ros::Duration &period,
                                    const ros::TimerCallback& callback);

private:
    std::map<boost::uuids::uuid, ros::Timer> timer_map;
};

class AsyncTriggerTimer {
private:
    ros::NodeHandlePtr nhp;
    std::mutex         mutex;
    ros::Publisher     bus_pub;
    ros::Publisher     spoof_evt_pub;
    ros::ServiceServer srvTrigger;
    bool               trigger_is_running = false;
    const double       default_freq       = 0.5;
    ros::Time          last_call          = ros::Time::now();
    ros::Time          next_expected      = ros::Time::now();
    ros::Duration      last_duration      = ros::Duration(0.5);
    ros::Duration      period_            = ros::Duration(1); /// this is the new main variable
    ros::Duration      min_period_        = ros::Duration(0.1); /// minimum time between triggers
    ros::Duration      max_period_        = ros::Duration(10.0); /// minimum time between triggers
    int                spoof_events_      = 0;
    std::shared_ptr<RedisEnvoy> envoy_;
    int                flippy             = 0;

    ros::TimerCallback callback{cb_print};
    OneShotManager osm;

public:

    AsyncTriggerTimer(ros::NodeHandlePtr nhp, ros::Duration period,
                      ros::Duration min_period, ros::Duration max_period,
                      int spoof_events, std::shared_ptr<RedisEnvoy> envoy);
    AsyncTriggerTimer(ros::NodeHandlePtr nhp, ros::Duration period);
    bool is_running();

    void start();
    void set_trigger_run(bool state);
    void cb_set_trigger_run(const std_msgs::Bool::ConstPtr &msg);

    /** This is merely a convenience wrapper around setPeriod
     *
     * @param frequency - Set the trigger frequency
     */
    void setRate(double frequency);
    void cb_setRate(const std_msgs::Float64::ConstPtr &msg);

    /** All timing sets should happen through here
     *
     * @param period - Set the trigger period, clipping to the min/max period
     */
    void setPeriod(const ros::Duration &period);
    void cb_setPeriod(const std_msgs::Float64::ConstPtr &msg);

    void setCallback(const ros::TimerCallback &callback_);

    void callTick(const ros::TimerEvent &event);

    void call();

    void call(const ros::TimerEvent &event);

    bool setTriggerRate(custom_msgs::SetTriggerRate::Request &req,
                        custom_msgs::SetTriggerRate::Response &resp);

    ros::Rate get_trigger_freq();
    ros::Duration get_trigger_dur();

};

class DaqWrapper {
private:
    ros::NodeHandlePtr node;
    UsbDaq             usbDaq;
    ros::Publisher     bus_pub;
    ros::ServiceServer srvTrigger;

public:

    DaqWrapper(ros::NodeHandlePtr node_, UsbDaq &usbDaq1);

    bool call();

    bool readPin( custom_msgs::ReadPinRequest &req,
                  custom_msgs::ReadPinResponse &rsp);
    bool analogWrite( custom_msgs::ReadPinRequest &req,
                        custom_msgs::ReadPinResponse &rsp);

};


#endif //MCC_DAQ_DAQ_NODE_H
