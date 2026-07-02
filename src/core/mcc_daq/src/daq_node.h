#ifndef MCC_DAQ_DAQ_NODE_H
#define MCC_DAQ_DAQ_NODE_H

#include <memory>
#include <mutex>
#include <map>
#include <functional>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64.hpp>

#include <roskv/envoy.h>
#include <custom_msgs/srv/set_trigger_rate.hpp>
#include <custom_msgs/srv/read_pin.hpp>
#include <custom_msgs/msg/gsof_evt.hpp>

#include "usbdaq.h"

using std::shared_ptr;

bool is_simple_frac(double d);

/// Minimal stand-in for ros::TimerEvent (rclcpp timers have void callbacks)
struct TimerEvent {
    rclcpp::Time current_real;
};

using TimerCallback = std::function<void(const TimerEvent &)>;

class TriggerTimer {
private:
    rclcpp::Node::SharedPtr node;
    UsbDaq             usbDaq;
    rclcpp::Service<custom_msgs::srv::SetTriggerRate>::SharedPtr srvTrigger;
    bool               trigger_is_running = false;
    const double       default_freq       = 0.5;
    const double       quick_idle_freq    = 10;
    double             freq_set           = default_freq;
    rclcpp::Time       last_call;
    rclcpp::Time       last_edge;
    rclcpp::Time       next_edge;
public:

    TriggerTimer(rclcpp::Node::SharedPtr node_, UsbDaq &usbDaq1);

    bool is_running();

    void set_trigger_run(bool state);

    void set_trigger_run(const std_msgs::msg::Bool::ConstSharedPtr &msg);

    void set_trigger_freq(double frequency);

    void set_trigger_freq(const std_msgs::msg::Float64::ConstSharedPtr &msg);
    rclcpp::Duration get_trigger_dur();

    void set_trigger_period(double t_seconds);

    void set_trigger_period(rclcpp::Duration duration);

    void setTriggerRate(const std::shared_ptr<custom_msgs::srv::SetTriggerRate::Request> req,
                        std::shared_ptr<custom_msgs::srv::SetTriggerRate::Response> resp);

    rclcpp::Time get_next_edge();
    void next();
    void spin_then_sleep();
    void sleep_until_edge(double granularity);

    /// current period, seconds
    double period_sec = 1.0 / 10;
};

class OneShotManager {
public:
    OneShotManager() = default;

    void erase(uint64_t i);

    uint64_t addOneShot(rclcpp::Node::SharedPtr nhp,
                        const rclcpp::Duration &period,
                        const TimerCallback& callback);

private:
    uint64_t next_id_ = 0;
    std::map<uint64_t, rclcpp::TimerBase::SharedPtr> timer_map;
};

class AsyncTriggerTimer {
private:
    rclcpp::Node::SharedPtr nhp;
    std::mutex         mutex;
    rclcpp::Publisher<custom_msgs::msg::GsofEvt>::SharedPtr spoof_evt_pub;
    rclcpp::Time       last_call;
    rclcpp::Time       next_expected;
    rclcpp::Duration   period_            = rclcpp::Duration::from_seconds(1); /// this is the new main variable
    rclcpp::Duration   min_period_        = rclcpp::Duration::from_seconds(0.1); /// minimum time between triggers
    rclcpp::Duration   max_period_        = rclcpp::Duration::from_seconds(10.0); /// maximum time between triggers
    int                spoof_events_      = 0;
    std::shared_ptr<RedisEnvoy> envoy_;

    TimerCallback callback;
    OneShotManager osm;

public:

    AsyncTriggerTimer(rclcpp::Node::SharedPtr nhp, rclcpp::Duration period,
                      rclcpp::Duration min_period, rclcpp::Duration max_period,
                      int spoof_events, std::shared_ptr<RedisEnvoy> envoy);
    AsyncTriggerTimer(rclcpp::Node::SharedPtr nhp, rclcpp::Duration period);

    void start();

    /** This is merely a convenience wrapper around setPeriod
     *
     * @param frequency - Set the trigger frequency
     */
    void setRate(double frequency);
    void cb_setRate(const std_msgs::msg::Float64::ConstSharedPtr &msg);

    /** All timing sets should happen through here
     *
     * @param period - Set the trigger period, clipping to the min/max period
     */
    void setPeriod(const rclcpp::Duration &period);
    void cb_setPeriod(const std_msgs::msg::Float64::ConstSharedPtr &msg);

    void setCallback(const TimerCallback &callback_);

    void callTick(const TimerEvent &event);

    void call();

    void call(const TimerEvent &event);

    rclcpp::Duration get_trigger_dur();

};

class DaqWrapper {
private:
    rclcpp::Node::SharedPtr node;
    UsbDaq             usbDaq;
    rclcpp::Service<custom_msgs::srv::ReadPin>::SharedPtr readPinSrv;

public:

    DaqWrapper(rclcpp::Node::SharedPtr node_, UsbDaq &usbDaq1);

    void readPin(const std::shared_ptr<custom_msgs::srv::ReadPin::Request> req,
                 std::shared_ptr<custom_msgs::srv::ReadPin::Response> rsp);

};


#endif //MCC_DAQ_DAQ_NODE_H
