#include <cstdio>
#include <cstring>
#include <chrono>

#include <stdio.h>

#include <ros/ros.h>
#include <ros/rate.h>
#include <std_msgs/String.h>
#include <std_msgs/Header.h>

#include <roskv/envoy.h>
#include <custom_msgs/SetTriggerRate.h>
#include <custom_msgs/ReadPin.h>
#include <custom_msgs/Stat.h>
#include <custom_msgs/GSOF_EVT.h>


#include "usb-2408.h"
#include "usbdaq.h"
#include "daq_node.h"
#include "utils.h"

using std::shared_ptr;

uint8_t G_INFO_VERBOSITY = 2;
static const bool TRUE = true;

/*
   Hello CPP
 */
#include <iostream>
#include<ros/ros.h>


ros::Time calculate_edge(ros::Duration const & granularity, ros::Duration const & thresh, ros::Duration const & offset);
ros::Time calculate_edge(ros::Duration const & granularity, ros::Duration const & thresh);
ros::Time calculate_edge(ros::Duration const & granularity);
std::string to_string(ros::Time const &);
std::string to_string(ros::Duration const &);
std::string to_string(ros::Rate const &);
int64_t to_nano64(ros::Duration const &);



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

std::string to_string(ros::Time const & t) {
    return std::to_string(t.toSec());
}
std::string to_string(ros::Duration const & t) {
    return std::to_string(t.toSec());
}
std::string to_string(ros::Rate const & t) {
    return std::to_string(ros::Duration(t).toSec());
}

ros::Time calculate_edge(ros::Duration const & granularity) {
    return calculate_edge(granularity, ros::Duration{0,0}, ros::Duration{0,0});
}

ros::Time calculate_edge(ros::Duration const & granularity, ros::Duration const & thresh) {
    return calculate_edge(granularity, thresh, ros::Duration{0,0});
}

ros::Time calculate_edge(ros::Duration const & granularity, ros::Duration const & thresh, ros::Duration const & offset) {
    auto now = ros::Time::now();
    auto nowns = now.toNSec();
    auto dt = granularity.toNSec();
    auto m = nowns % dt;
    auto delay_ns = dt - m;
    ros::Duration delay = ros::Duration().fromNSec(delay_ns) - offset;
//  ROS_INFO("Now: %s", to_string(now).c_str());
//  ROS_INFO("dt: %s", to_string(granularity).c_str());
//  ROS_INFO("delay_ns: %ld", delay_ns);
//    ROS_INFO("delaydur: %s", to_string(out).c_str());
    if (delay < thresh) {
        delay = ros::Duration{0,0};
    }
    ros::Time edge = now + delay;
    return edge;

}

void chatterCallback(const std_msgs::String::ConstPtr& msg) {
    ROS_INFO("I heard: [%s]", msg->data.c_str());
}

TriggerTimer::TriggerTimer(ros::NodeHandlePtr node_, UsbDaq &usbDaq_) {
    node = node_;
    usbDaq = usbDaq_;
    srvTrigger = node->advertiseService("set_trigger_rate", &TriggerTimer::setTriggerRate, this);
}

void TriggerTimer::set_trigger_run(bool state) {
    if (state) {
        ROS_INFO1("Turned on timer")
//        timer.start();
        trigger_is_running = true;
        rate = ros::Rate(freq_set);
    } else {
        ROS_INFO1("Turned off timer")
//        timer.stop();
        trigger_is_running = false;
        rate = ros::Rate(quick_idle_freq);
    }
}

void TriggerTimer::set_trigger_run(const std_msgs::Bool::ConstPtr &msg) {
    set_trigger_run(msg->data);
}

/** Set trigger timer duration
 * If set to 0 or less, disable the timer.
 * @param duration
 */
void TriggerTimer::set_trigger_period(ros::Duration duration) {
    if (duration.toSec() > 0.0) {
        ROS_INFO1("Start timer")
//        timer.setPeriod(duration);
//        timer.start();
        set_trigger_run(true);
    } else {
        ROS_INFO1("Stop timer")
//        timer.stop();
        set_trigger_run(false);
    }
}

void TriggerTimer::set_trigger_period(double t_seconds) {
    set_trigger_period(ros::Duration(t_seconds));
}

void TriggerTimer::set_trigger_freq(double frequency) {
    ROS_INFO("Setting trigger freq %2.3f hz", frequency);

    if (frequency > 0) {
        freq_set = frequency;
        rate = ros::Rate(freq_set);
    } else {
        rate = ros::Rate(quick_idle_freq);
    }

}

void TriggerTimer::set_trigger_freq(const std_msgs::Float64::ConstPtr &msg) {
    set_trigger_freq(msg->data);
}

ros::Rate TriggerTimer::get_trigger_freq() {
    return ros::Rate(rate);
}
ros::Duration TriggerTimer::get_trigger_dur() {
    return ros::Duration(rate);
}

bool TriggerTimer::is_running() {
    return trigger_is_running;
}

//bool TriggerTimer::call() {}

void TriggerTimer::nop(const ros::TimerEvent &event) {
}

bool TriggerTimer::setTriggerRate(custom_msgs::SetTriggerRate::Request &req,
                                  custom_msgs::SetTriggerRate::Response &resp) {
    set_trigger_freq(req.rate);
    resp.success = true;
    return true;
}

/// Compute the next timer edge based on the last set frequency
ros::Time TriggerTimer::get_next_edge() {
// todo: bounds checking here
// todo: put behind debugging env variable
    auto dur = get_trigger_dur();
    ros::Time next_edge;
    if (is_simple_frac(dur.toSec())) {
        ROS_INFO("simple dur: %s", to_string(dur).c_str());
        next_edge = calculate_edge(dur, ros::Duration(0.01), ros::Duration(0));
    } else {
        next_edge = ros::Time::now() + dur;
        ROS_WARN("complex dur: %s", to_string(dur).c_str());
    }
    return next_edge;
}

/// Advance the timer
void TriggerTimer::next() {
    auto now = ros::Time::now();
    last_edge = next_edge;
    next_edge = get_next_edge();
    ROS_INFO("Last: %s Next: %s Last Call: %s Now: %s dt: %s", to_string(last_edge).c_str(), to_string(next_edge).c_str(),
        to_string(last_call).c_str(), to_string(now).c_str(), to_string(now-last_call).c_str());
    last_call = now;
}

/// sleep until the next edge. this is mostly a convenience method.
void TriggerTimer::sleep_until_edge(double granularity) {
    auto next_edge = calculate_edge(ros::Duration(granularity), ros::Duration(0.01), ros::Duration(0.0001));
    auto till_next_edge = next_edge - ros::Time::now();
    ROS_WARN("Next edge: %s", to_string(till_next_edge).c_str());
    till_next_edge.sleep();
}


/// spin once then sleep until the next period starts
void TriggerTimer::spin_then_sleep() {
    next();
    ros::spinOnce();
    auto till_next_edge = next_edge - ros::Time::now();
    till_next_edge.sleep();
}

//
void TriggerTimer::sleep_next() {

}
/** =================== AsyncTriggerTimer ============================== */

void OneShotManager::erase(boost::uuids::uuid i) {
    timer_map.erase(i);
//    std::cout << "erasing: " << i << " sz: " << timer_map.size() <<std::endl;
}

boost::uuids::uuid OneShotManager::addOneShot(ros::NodeHandlePtr nhp,
        const ros::Duration &period,
        const ros::TimerCallback& callback) {
    boost::uuids::uuid i = boost::uuids::random_generator()();

    ros::TimerCallback cb2 = [i, this, callback](const ros::TimerEvent &e) {
        callback(e);
        erase(i);
    };
    ros::Timer tmp = nhp->createTimer(period, cb2, true, false);
    timer_map.emplace(i, tmp);
    tmp.start(); // safety here, need to ensure it's in the map before it pops
    return i;
}


AsyncTriggerTimer::AsyncTriggerTimer(ros::NodeHandlePtr nhp, ros::Duration period,
                                     ros::Duration min_period, ros::Duration max_period,
                                     int spoof_events, std::shared_ptr<RedisEnvoy> envoy)
: nhp{nhp}, period_{period}, min_period_{min_period}, max_period_{max_period}, spoof_events_{spoof_events}, envoy_{envoy}{
      spoof_evt_pub  = nhp->advertise<custom_msgs::GSOF_EVT> ("/event", 1);
}
AsyncTriggerTimer::AsyncTriggerTimer(ros::NodeHandlePtr nhp, ros::Duration period)
: nhp{nhp}, period_{period} {}

void AsyncTriggerTimer::start() {
    callTick(ros::TimerEvent{});
}

void AsyncTriggerTimer::setPeriod(const ros::Duration &period) {
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
    auto period = ros::Duration(ros::Rate(rate));
    setPeriod(period);
}

void AsyncTriggerTimer::cb_setPeriod(const std_msgs::Float64::ConstPtr &msg) {
    ROS_INFO("& Got double: %lf, set period", msg->data);
    setPeriod(ros::Duration(msg->data));
}
void AsyncTriggerTimer::cb_setRate(const std_msgs::Float64::ConstPtr &msg) {
    ROS_INFO("& Got double: %lf, set rate", msg->data);
    setRate(msg->data);
}

void AsyncTriggerTimer::call() {
    ros::TimerEvent event;
    event.current_real = ros::Time::now();
    call(event);
}

/// Call the bound callback
void AsyncTriggerTimer::call(const ros::TimerEvent &e) {
    auto now = ros::Time::now();
    ROS_INFO("! AdjT RealDT( %lf )", (now - last_call).toSec());
    spoof_events_ = get_redis_int(envoy_, "/debug/spoof_events");
    if (spoof_events_ == 1) {
        // We're going to spoof a GSOF_EVT, so we don't depend on the
        // INS always having a good sync to test the system
        ROS_WARN("Spoofing GSOF_EVT message!");
        custom_msgs::GSOF_EVT msg;
        msg.gps_time = now + ros::Duration(1e-4); // Add some small amount of noise to differ gps from sys
        msg.sys_time = now;
        msg.time = now.toSec();
        msg.header.stamp = now + ros::Duration(1e-4); // Header should match gps time
        spoof_evt_pub.publish(msg);
    }
    callback(e);
    last_call = now;
}


/// Bind a callback
void AsyncTriggerTimer::setCallback(const ros::TimerCallback& callback_) {
    callback = callback_;
}

/// This runs the mutually recursive loop. Enqueue the next event, then
/// call the callback
void AsyncTriggerTimer::callTick(const ros::TimerEvent &e) {
    std::lock_guard<std::mutex> guard(mutex);
    /// "now" is actually event.current_real
//    ROS_INFO("%d AdjT %s dt: %lf period: %3.2lf Evt: %s",i++, isonow().c_str(), (e.last_real - last).toSec(), period_.toSec(), to_string(e).c_str());
    next_expected = e.current_real + period_;
    ros::Duration nextPeriod = next_expected - ros::Time::now();
//        ROS_INFO("now: %lf last: %lf next: %lf dur: %lf", e.last_real.toSec(), last.toSec(), nextExpected.toSec(), dur.toSec());

    if (ros::ok()) {
//        ROS_INFO("enqueuing next callTick");
        ros::TimerCallback nextCycle = boost::bind(&AsyncTriggerTimer::callTick, this, _1);
        osm.addOneShot(nhp, period_, nextCycle);
    }

    call(e);
}

ros::Rate AsyncTriggerTimer::get_trigger_freq() {
    return ros::Rate(period_);
}
ros::Duration AsyncTriggerTimer::get_trigger_dur() {
    return ros::Duration(period_);
}


/** ===================  daq wrapper ============================== */

DaqWrapper::DaqWrapper(ros::NodeHandlePtr node_, UsbDaq &usbDaq_) {
    node = node_;
    usbDaq = usbDaq_;
}

bool DaqWrapper::readPin(custom_msgs::ReadPinRequest &req,
                           custom_msgs::ReadPinResponse &rsp) {
    ROS_INFO("Starting to read on pin %d", req.pin);
    rsp.value = usbDaq.voltageRead((uint8_t) req.pin, BP_10V);
    return true;
}

/** ===================  main ============================== */
int main(int argc, char** argv) {
    RedisEnvoyOpts envoy_opts = RedisEnvoyOpts::from_env("daq" );
    auto envoy_ = std::make_shared<RedisEnvoy>(envoy_opts);
    ROS_INFO("echo: %s", envoy_->echo("Redis connected").c_str());


    int debug = 0;
    int trigger_pps = 0;
    // Enables publishing a GSOF_EVT on each pulse
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
    double spoof_rate = 0;
    double spoof_daq = false;
    ros::init(argc, argv, "daq_node");
//    ros::NodeHandle nh;
    ros::NodeHandle nh;
    ros::NodeHandlePtr node = boost::make_shared<ros::NodeHandle>(nh);
    ros::Publisher     bus_pub  = node->advertise<std_msgs::String> ("/bus", 10);
    ros::Publisher     stat_pub = node->advertise<custom_msgs::Stat>("/stat", 3);
    ros::Publisher     trig_pub = node->advertise<std_msgs::Header>("/trig", 3);

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

    bool start_running = true; // Trigger starts with node start
    bool dummy_mode = false; // Use the dummy daq code
    node->param("start_running", start_running, TRUE);
    node->param("dummy", dummy_mode, false);
    ros::Duration(0, 500000000).sleep(); // sleep briefly to give the publisher time to catch up
    ros::param::get("/spoof_rate", spoof_rate);
    ros::param::get("/spoof_daq", spoof_daq);

    if (spoof_rate > 0) {
        ROS_WARN("\nGoing into spoof mode \n");
        ros::spin();
        return 0;
    }

//    std::shared_ptr<UsbDaqBase> daq;
//    if (dummy_mode) {
//        daq = std::shared_ptr<UsbDaqDummy>(node);
//    } else {
//        daq = new UsbDaq(node);
//    }
    auto *daq = new UsbDaq(node);
    TriggerTimer triggerTimer = TriggerTimer(node, *daq);
    auto asyncTriggerTimerP = std::make_shared<AsyncTriggerTimer>(node,
                                        ros::Duration(1.0),
                                        ros::Duration(min_period),
                                        ros::Duration(max_period),
                                        spoof_events,
                                        envoy_);
    DaqWrapper daqWrapper = DaqWrapper(node, *daq);
    daq->init();
    daq->digitalPulse();


    std_msgs::String msg;
    msg.data = "~~~~~ DAQ node going online ~~~~~";
    bus_pub.publish(msg);

//    ros::Subscriber sub                 =  node->subscribe(node_ns+"/"+topic, 10, chatterCallback);
    ros::Subscriber sub_blip_period     =  node->subscribe(node_ns+"/blip_period", 10, &UsbDaq::set_blip_micros, daq);
    ros::Subscriber sub_blipper         =  node->subscribe(node_ns+"/"+topic, 10, &UsbDaq::switchboard, daq);
    ros::Subscriber sub_pulser          =  node->subscribe(node_ns+"/pulse", 10, &UsbDaq::pulse, daq);
    ros::Subscriber sub_trigger_freq    =  node->subscribe(node_ns+"/trigger_freq", 10,
                                                         &AsyncTriggerTimer::cb_setRate, &(*asyncTriggerTimerP));
    ros::Subscriber sub_trigger_run     =  node->subscribe(node_ns+"/trigger_run", 10,
                                                         &TriggerTimer::set_trigger_run, &triggerTimer);
    ros::ServiceServer read_pin_srv_    =  node->advertiseService(node_ns+"/read_pin",
                                                         &DaqWrapper::readPin, &daqWrapper) ;

    double trigger_freq = 0.5;
    node->param<double>("/trigger_freq", trigger_freq, 0.5);


    auto nodeName = ros::this_node::getName();
    custom_msgs::Stat stat_msg;
    stat_msg.node = nodeName;
    stat_msg.trace_topic = nodeName + "/blip";

    uint32_t counter = 0;
    triggerTimer.set_trigger_run(start_running);
    triggerTimer.set_trigger_freq(trigger_freq);

    ros::Time next_edge;
    ros::Duration till_next_edge;
    triggerTimer.sleep_until_edge(1.0);

    asyncTriggerTimerP->setCallback([daq, envoy_, asyncTriggerTimerP, trig_pub](const ros::TimerEvent &event) {
        std_msgs::Header header;
        header.stamp = ros::Time::now();
        trig_pub.publish(header);
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
            while (ros::ok()) {
        //        ros::spinOnce();
                if (triggerTimer.is_running()) {
                    std_msgs::Header header;
                    header.stamp = ros::Time::now();
                    header.frame_id = "1.0";
                    trig_pub.publish(header);
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
    ros::AsyncSpinner spinner{0};
    asyncTriggerTimerP->start();
    spinner.start();
    ros::waitForShutdown();

    return 0;
    /// DEPRECATED

}
