#include <cstdio>
#include <cstring>
#include <chrono>
#include <cmath>
#include <iostream>

#include <stdio.h>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/u_int32.hpp>

#include <roskv/envoy.h>
#include <custom_msgs/srv/set_trigger_rate.hpp>
#include <custom_msgs/srv/read_pin.hpp>
#include <custom_msgs/msg/stat.hpp>
#include <custom_msgs/msg/gsof_evt.hpp>


#include "usb-2408.h"
#include "usbdaq.h"
#include "daq_node.h"
#include "utils.h"

using std::shared_ptr;
using namespace std::chrono_literals;

uint8_t G_INFO_VERBOSITY = 2;

static rclcpp::Node::SharedPtr g_node;

static rclcpp::Time now_() {
    return g_node->now();
}

rclcpp::Time calculate_edge(rclcpp::Duration const & granularity, rclcpp::Duration const & thresh, rclcpp::Duration const & offset);
std::string to_string(rclcpp::Time const &);
std::string to_string(rclcpp::Duration const &);


/** ===================  helper func ============================== */
double get_redis_double(std::shared_ptr<RedisEnvoy> envoy, const std::string &key) {
        std::string tmp = envoy->get(key);
        json::json tmpj = json::json::parse(tmp);
        if (!tmpj.is_primitive() || tmpj.is_null()) {
            throw std::invalid_argument("Redis Key " + key + " must exist and be of primitive type");
        }
        if (tmpj.is_number_integer()) {
            return (double) tmpj.front();
        } else if (tmpj.is_number_float()) {
            return (double) tmpj.front();
        }
        throw std::invalid_argument("Redis Key " + key + " is invalid type");
    }

int get_redis_int(std::shared_ptr<RedisEnvoy> envoy, const std::string &key) {
        std::string tmp = envoy->get(key);
        json::json tmpj = json::json::parse(tmp);
        if (!tmpj.is_primitive() || tmpj.is_null()) {
            throw std::invalid_argument("Redis Key " + key + " must exist and be of primitive type");
        }
        if (tmpj.is_number_integer()) {
            return (int) tmpj.front();
        } else if (tmpj.is_boolean()) {
            bool b = tmpj.front();
            return (int) b;
        } else if (tmpj.is_number_float()) {
            double x = tmpj.front();
            return x != 0;
        }
        throw std::invalid_argument("Redis Key " + key + " is invalid type");
}
/// Determine if a number is a "simple fraction". A simple fraction is one that is either a whole number or a
/// fraction with a small denominator. This ensures that pulses only edge-align if there is a clean, round modulus.
/// This is primarily for debugging and should be reduced to 1/1 ratio in production.
bool is_simple_frac(double d) {
    const double ratio = 1.0/12; // ratio considered for determining what is a "simple" fraction
    auto m = fmod(d, ratio);
    return m < 1e-9;
}

std::string to_string(rclcpp::Time const & t) {
    return std::to_string(t.seconds());
}
std::string to_string(rclcpp::Duration const & t) {
    return std::to_string(t.seconds());
}

rclcpp::Time calculate_edge(rclcpp::Duration const & granularity, rclcpp::Duration const & thresh, rclcpp::Duration const & offset) {
    auto now = now_();
    auto nowns = now.nanoseconds();
    auto dt = granularity.nanoseconds();
    auto m = nowns % dt;
    auto delay_ns = dt - m;
    rclcpp::Duration delay = rclcpp::Duration(std::chrono::nanoseconds(delay_ns)) - offset;
    if (delay < thresh) {
        delay = rclcpp::Duration(0, 0);
    }
    rclcpp::Time edge = now + delay;
    return edge;
}

TriggerTimer::TriggerTimer(rclcpp::Node::SharedPtr node_, UsbDaq &usbDaq_)
    : last_call(node_->now()), last_edge(node_->now()), next_edge(node_->now()) {
    node = node_;
    usbDaq = usbDaq_;
    srvTrigger = node->create_service<custom_msgs::srv::SetTriggerRate>(
        "set_trigger_rate",
        std::bind(&TriggerTimer::setTriggerRate, this, std::placeholders::_1, std::placeholders::_2));
}

void TriggerTimer::set_trigger_run(bool state) {
    if (state) {
        ROS_INFO1("Turned on timer")
        trigger_is_running = true;
        period_sec = 1.0 / freq_set;
    } else {
        ROS_INFO1("Turned off timer")
        trigger_is_running = false;
        period_sec = 1.0 / quick_idle_freq;
    }
}

void TriggerTimer::set_trigger_run(const std_msgs::msg::Bool::ConstSharedPtr &msg) {
    set_trigger_run(msg->data);
}

/** Set trigger timer duration
 * If set to 0 or less, disable the timer.
 * @param duration
 */
void TriggerTimer::set_trigger_period(rclcpp::Duration duration) {
    if (duration.seconds() > 0.0) {
        ROS_INFO1("Start timer")
        set_trigger_run(true);
    } else {
        ROS_INFO1("Stop timer")
        set_trigger_run(false);
    }
}

void TriggerTimer::set_trigger_period(double t_seconds) {
    set_trigger_period(rclcpp::Duration::from_seconds(t_seconds));
}

void TriggerTimer::set_trigger_freq(double frequency) {
    ROS_INFO("Setting trigger freq %2.3f hz", frequency);

    if (frequency > 0) {
        freq_set = frequency;
        period_sec = 1.0 / freq_set;
    } else {
        period_sec = 1.0 / quick_idle_freq;
    }

}

void TriggerTimer::set_trigger_freq(const std_msgs::msg::Float64::ConstSharedPtr &msg) {
    set_trigger_freq(msg->data);
}

rclcpp::Duration TriggerTimer::get_trigger_dur() {
    return rclcpp::Duration::from_seconds(period_sec);
}

bool TriggerTimer::is_running() {
    return trigger_is_running;
}

void TriggerTimer::setTriggerRate(const std::shared_ptr<custom_msgs::srv::SetTriggerRate::Request> req,
                                  std::shared_ptr<custom_msgs::srv::SetTriggerRate::Response> resp) {
    set_trigger_freq(req->rate);
    resp->success = true;
}

/// Compute the next timer edge based on the last set frequency
rclcpp::Time TriggerTimer::get_next_edge() {
// todo: bounds checking here
// todo: put behind debugging env variable
    auto dur = get_trigger_dur();
    rclcpp::Time next_edge_;
    if (is_simple_frac(dur.seconds())) {
        ROS_INFO("simple dur: %s", to_string(dur).c_str());
        next_edge_ = calculate_edge(dur, rclcpp::Duration::from_seconds(0.01), rclcpp::Duration(0, 0));
    } else {
        next_edge_ = now_() + dur;
        ROS_WARN("complex dur: %s", to_string(dur).c_str());
    }
    return next_edge_;
}

/// Advance the timer
void TriggerTimer::next() {
    auto now = now_();
    last_edge = next_edge;
    next_edge = get_next_edge();
    ROS_INFO("Last: %s Next: %s Last Call: %s Now: %s dt: %s", to_string(last_edge).c_str(), to_string(next_edge).c_str(),
        to_string(last_call).c_str(), to_string(now).c_str(), to_string(now-last_call).c_str());
    last_call = now;
}

/// sleep until the next edge. this is mostly a convenience method.
void TriggerTimer::sleep_until_edge(double granularity) {
    auto edge = calculate_edge(rclcpp::Duration::from_seconds(granularity),
                               rclcpp::Duration::from_seconds(0.01),
                               rclcpp::Duration::from_seconds(0.0001));
    auto till_next_edge = edge - now_();
    ROS_WARN("Next edge: %s", to_string(till_next_edge).c_str());
    if (till_next_edge.nanoseconds() > 0) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(till_next_edge.nanoseconds()));
    }
}


/// spin once then sleep until the next period starts
void TriggerTimer::spin_then_sleep() {
    next();
    rclcpp::spin_some(node);
    auto till_next_edge = next_edge - now_();
    if (till_next_edge.nanoseconds() > 0) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(till_next_edge.nanoseconds()));
    }
}

/** =================== AsyncTriggerTimer ============================== */

void OneShotManager::erase(uint64_t i) {
    timer_map.erase(i);
}

uint64_t OneShotManager::addOneShot(rclcpp::Node::SharedPtr nhp,
        const rclcpp::Duration &period,
        const TimerCallback& callback) {
    uint64_t i = next_id_++;

    auto cb2 = [i, this, callback, nhp]() {
        auto it = timer_map.find(i);
        if (it != timer_map.end()) {
            it->second->cancel();
        }
        TimerEvent e;
        e.current_real = nhp->now();
        callback(e);
        erase(i);
    };
    rclcpp::TimerBase::SharedPtr tmp = nhp->create_wall_timer(
        std::chrono::nanoseconds(period.nanoseconds()), cb2);
    timer_map.emplace(i, tmp);
    return i;
}


AsyncTriggerTimer::AsyncTriggerTimer(rclcpp::Node::SharedPtr nhp, rclcpp::Duration period,
                                     rclcpp::Duration min_period, rclcpp::Duration max_period,
                                     int spoof_events, std::shared_ptr<RedisEnvoy> envoy)
: nhp{nhp}, last_call{nhp->now()}, next_expected{nhp->now()},
  period_{period}, min_period_{min_period}, max_period_{max_period},
  spoof_events_{spoof_events}, envoy_{envoy} {
      spoof_evt_pub = nhp->create_publisher<custom_msgs::msg::GsofEvt>("/event", 1);
}
AsyncTriggerTimer::AsyncTriggerTimer(rclcpp::Node::SharedPtr nhp, rclcpp::Duration period)
: nhp{nhp}, last_call{nhp->now()}, next_expected{nhp->now()}, period_{period} {}

void AsyncTriggerTimer::start() {
    TimerEvent e;
    e.current_real = nhp->now();
    callTick(e);
}

void AsyncTriggerTimer::setPeriod(const rclcpp::Duration &period) {
    if (period == period_) return;
    if (period < min_period_) {
        period_ = min_period_;
    } else if (period > max_period_) {
        period_ = max_period_;
    } else {
        period_ = period;
    }
}

void AsyncTriggerTimer::setRate(double rate) {
    setPeriod(rclcpp::Duration::from_seconds(1.0 / rate));
}

void AsyncTriggerTimer::cb_setPeriod(const std_msgs::msg::Float64::ConstSharedPtr &msg) {
    ROS_INFO("& Got double: %lf, set period", msg->data);
    setPeriod(rclcpp::Duration::from_seconds(msg->data));
}
void AsyncTriggerTimer::cb_setRate(const std_msgs::msg::Float64::ConstSharedPtr &msg) {
    ROS_INFO("& Got double: %lf, set rate", msg->data);
    setRate(msg->data);
}

void AsyncTriggerTimer::call() {
    TimerEvent event;
    event.current_real = nhp->now();
    call(event);
}

/// Call the bound callback
void AsyncTriggerTimer::call(const TimerEvent &e) {
    auto now = nhp->now();
    ROS_INFO("! AdjT RealDT( %lf )", (now - last_call).seconds());
    spoof_events_ = get_redis_int(envoy_, "/debug/spoof_events");
    if (spoof_events_ == 1) {
        // We're going to spoof a GsofEvt, so we don't depend on the
        // INS always having a good sync to test the system
        ROS_WARN("Spoofing GsofEvt message!");
        custom_msgs::msg::GsofEvt msg;
        auto gps_time = now + rclcpp::Duration::from_seconds(1e-4); // Add some small amount of noise to differ gps from sys
        msg.gps_time = gps_time;
        msg.sys_time = now;
        msg.time = now.seconds();
        msg.header.stamp = gps_time; // Header should match gps time
        spoof_evt_pub->publish(msg);
    }
    if (callback) {
        callback(e);
    }
    last_call = now;
}


/// Bind a callback
void AsyncTriggerTimer::setCallback(const TimerCallback& callback_) {
    callback = callback_;
}

/// This runs the mutually recursive loop. Enqueue the next event, then
/// call the callback
void AsyncTriggerTimer::callTick(const TimerEvent &e) {
    std::lock_guard<std::mutex> guard(mutex);
    /// "now" is actually event.current_real
    next_expected = e.current_real + period_;

    if (rclcpp::ok()) {
        TimerCallback nextCycle = [this](const TimerEvent &ev) { callTick(ev); };
        osm.addOneShot(nhp, period_, nextCycle);
    }

    call(e);
}

rclcpp::Duration AsyncTriggerTimer::get_trigger_dur() {
    return period_;
}


/** ===================  daq wrapper ============================== */

DaqWrapper::DaqWrapper(rclcpp::Node::SharedPtr node_, UsbDaq &usbDaq_) {
    node = node_;
    usbDaq = usbDaq_;
    readPinSrv = node->create_service<custom_msgs::srv::ReadPin>(
        "/daq/read_pin",
        std::bind(&DaqWrapper::readPin, this, std::placeholders::_1, std::placeholders::_2));
}

void DaqWrapper::readPin(const std::shared_ptr<custom_msgs::srv::ReadPin::Request> req,
                         std::shared_ptr<custom_msgs::srv::ReadPin::Response> rsp) {
    ROS_INFO("Starting to read on pin %d", req->pin);
    rsp->value = usbDaq.voltageRead((uint8_t) req->pin, BP_10V);
}

/** ===================  main ============================== */
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("daq_node");
    g_node = node;

    RedisEnvoyOpts envoy_opts = RedisEnvoyOpts::from_env("daq" );
    auto envoy_ = std::make_shared<RedisEnvoy>(envoy_opts);
    ROS_INFO("echo: %s", envoy_->echo("Redis connected").c_str());


    int debug = 0;
    int trigger_pps = 0;
    // Enables publishing a GsofEvt on each pulse
    // (not reliant on INS having a good sync)
    int spoof_events = 0;
    try {
        debug = get_redis_int(envoy_, "/debug/enable");
        trigger_pps = get_redis_int(envoy_, "/debug/trigger_pps");
        spoof_events = get_redis_int(envoy_, "/debug/spoof_events");
        ROS_INFO_STREAM("Spoof events is: " << spoof_events);
    } catch (std::exception &e) {
        ROS_WARN("tried to get debug keys but failed: %s", e.what());
    }

    std::string node_ns = "/daq";
    std::string topic = "chatter";
    auto bus_pub  = node->create_publisher<std_msgs::msg::String>("/bus", 10);
    auto stat_pub = node->create_publisher<custom_msgs::msg::Stat>("/stat", 3);
    auto trig_pub = node->create_publisher<std_msgs::msg::Header>("/trig", 3);

    double min_period = 0.1;
    double max_period = 10.0;
    {
        const char *env_min_trig = std::getenv("TRIGGER_MIN_PERIOD");
        const char *env_max_trig = std::getenv("TRIGGER_MAX_PERIOD");
        if (env_min_trig && env_min_trig[0]) {
            min_period = std::stof(std::string(env_min_trig));
        }
        if (env_max_trig && env_max_trig[0]) {
            max_period = std::stof(std::string(env_max_trig));
        }
    }

    bool start_running = node->declare_parameter("start_running", true); // Trigger starts with node start
    node->declare_parameter("dummy", false); // Use the dummy daq code (currently unused)
    double spoof_rate = 0.0;
    {
        const char *env_spoof = std::getenv("SPOOF_RATE");
        if (env_spoof && env_spoof[0]) {
            spoof_rate = std::stof(std::string(env_spoof));
        }
    }

    if (spoof_rate > 0) {
        ROS_WARN("\nGoing into spoof mode \n");
        rclcpp::spin(node);
        return 0;
    }

    auto *daq = new UsbDaq(node);
    TriggerTimer triggerTimer = TriggerTimer(node, *daq);
    auto asyncTriggerTimerP = std::make_shared<AsyncTriggerTimer>(node,
                                        rclcpp::Duration::from_seconds(1.0),
                                        rclcpp::Duration::from_seconds(min_period),
                                        rclcpp::Duration::from_seconds(max_period),
                                        spoof_events,
                                        envoy_);
    DaqWrapper daqWrapper = DaqWrapper(node, *daq);
    daq->init();
    daq->digitalPulse();


    std_msgs::msg::String msg;
    msg.data = "~~~~~ DAQ node going online ~~~~~";
    bus_pub->publish(msg);

    auto sub_blip_period = node->create_subscription<std_msgs::msg::UInt32>(
        node_ns + "/blip_period", 10,
        [daq](const std_msgs::msg::UInt32::ConstSharedPtr m) { daq->set_blip_micros(m); });
    auto sub_blipper = node->create_subscription<std_msgs::msg::String>(
        node_ns + "/" + topic, 10,
        [daq](const std_msgs::msg::String::ConstSharedPtr m) { daq->switchboard(m); });
    auto sub_pulser = node->create_subscription<std_msgs::msg::UInt32>(
        node_ns + "/pulse", 10,
        [daq](const std_msgs::msg::UInt32::ConstSharedPtr m) { daq->pulse(m); });
    auto sub_trigger_freq = node->create_subscription<std_msgs::msg::Float64>(
        node_ns + "/trigger_freq", 10,
        [asyncTriggerTimerP](const std_msgs::msg::Float64::ConstSharedPtr m) {
            asyncTriggerTimerP->cb_setRate(m);
        });
    auto sub_trigger_run = node->create_subscription<std_msgs::msg::Bool>(
        node_ns + "/trigger_run", 10,
        [&triggerTimer](const std_msgs::msg::Bool::ConstSharedPtr m) {
            triggerTimer.set_trigger_run(m);
        });

    double trigger_freq = node->declare_parameter("trigger_freq", 0.5);


    auto nodeName = std::string(node->get_fully_qualified_name());
    custom_msgs::msg::Stat stat_msg;
    stat_msg.node = nodeName;
    stat_msg.trace_topic = nodeName + "/blip";

    uint32_t counter = 0;
    triggerTimer.set_trigger_run(start_running);
    triggerTimer.set_trigger_freq(trigger_freq);

    triggerTimer.sleep_until_edge(1.0);

    asyncTriggerTimerP->setCallback([daq, envoy_, asyncTriggerTimerP, trig_pub, node](const TimerEvent &event) {
        std_msgs::msg::Header header;
        header.stamp = node->now();
        trig_pub->publish(header);
        daq->blip();
        ROS_DEBUG("blip");
        ROS_INFO("checking redis");
        try {
            double new_trigger_freq = get_redis_double(envoy_, "/sys/arch/trigger_freq");
            asyncTriggerTimerP->setRate(new_trigger_freq);
        } catch (std::exception &e) {
            ROS_WARN("tried to set trigger from redis but failed: %s", e.what());
            throw;
        }
    });
    if (debug) {
        ROS_WARN("Debug enabled");
        if (trigger_pps) {
            ROS_WARN("PPS enabled");
            triggerTimer.set_trigger_freq(1.0);
            triggerTimer.sleep_until_edge(1.0);
            while (rclcpp::ok()) {
                if (triggerTimer.is_running()) {
                    std_msgs::msg::Header header;
                    header.stamp = node->now();
                    header.frame_id = "1.0";
                    trig_pub->publish(header);
                    daq->blip();
                } else {
                    if (G_INFO_VERBOSITY > 3) {
                        // sleep spinner
                        printf("\b%c", "|/-\\"[counter & 0x3]);
                        fflush(stdout);
                    }
                }
                triggerTimer.spin_then_sleep();
                counter++;
            } // end loop
            return 0;

        }
    }


    ROS_INFO("starting trigger seq");
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    asyncTriggerTimerP->start();
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
