#include <fstream>
#include <iostream>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <custom_msgs/msg/gsof_evt.hpp>
#include <cam_utils/event_cache.hpp>

static rclcpp::Logger LOG = rclcpp::get_logger("event_cache");

static bool stamp_is_valid(const builtin_interfaces::msg::Time &t) {
    return t.sec != 0 || t.nanosec != 0;
}


rclcpp::Duration rosabs(rclcpp::Duration dur) {
    if (dur < ZERO_DURATION)
        return rclcpp::Duration(std::chrono::nanoseconds(-dur.nanoseconds()));
    return dur;
}


EventCache::EventCache() {
    this->set_tolerance(rclcpp::Duration::from_seconds(0.499));
    this->set_delay(rclcpp::Duration(0, 0));
}

EventCache::EventCache(rclcpp::Duration tol, rclcpp::Duration delay) {
    this->set_tolerance(tol);
    this->set_delay(delay);
}

void EventCache::set_tolerance(rclcpp::Duration const &tolerance) {
    std::lock_guard<std::mutex> guard(mutex_);
    this->tol = tolerance;
}

void EventCache::set_delay(rclcpp::Duration const &delay) {
    std::lock_guard<std::mutex> guard(mutex_);
    this->delay = delay;
}

void EventCache::set_delay(double delay) {
    std::lock_guard<std::mutex> guard(mutex_);
    this->delay = rclcpp::Duration::from_seconds(delay);
}

void EventCache::push_back(rclcpp::Time const &t, const custom_msgs::msg::GsofEvt::ConstSharedPtr& msg) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!stamp_is_valid(msg->sys_time)) {
        RCLCPP_ERROR(LOG, "Zero/invalid sys_time encountered in EventCache::push_back()");
        return;
    }
    if (!stamp_is_valid(msg->gps_time)) {
        RCLCPP_ERROR(LOG, "Zero/invalid gps_time encountered in EventCache::push_back()");
        return;
    }
    event_map.emplace(t, custom_msgs::msg::GsofEvt(*msg));
}

int EventCache::size() {
    std::lock_guard<std::mutex> guard(mutex_);
    return (int) event_map.size();
}

void EventCache::show() {
    std::lock_guard<std::mutex> guard(mutex_);
    for (auto it = event_map.begin(); it != event_map.end(); ++it) {
        std::cout << it->second.header.stamp.sec << "." << it->second.header.stamp.nanosec
                  << " " << it->second.header.frame_id << "\n";
    }
    std::cout << "\n---" << std::endl;
}

bool EventCache::search(rclcpp::Time image_time, std_msgs::msg::Header &head,
                        uint64_t &event_num, bool remove_when_found) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (image_time.nanoseconds() == 0) {
        RCLCPP_ERROR(LOG, "zero/invalid image_time encountered in EventCache::search()");
        return false;
    }
    int count = 0;
    // we want the lowest corrected time
    rclcpp::Duration best_time{999, 999};
    bool have_best = false;
    rclcpp::Time best_sys_time; // best matching system time, aka key
    RCLCPP_INFO_STREAM(LOG, "Image time is: " << image_time.seconds());
    std::map<rclcpp::Time, custom_msgs::msg::GsofEvt> event_map_copy(event_map);
    for (const auto &pair: event_map_copy) {
        rclcpp::Time sys_time(pair.second.sys_time);
        auto actual_delay = rosabs(sys_time - image_time);
        auto corrected_delay = rosabs(actual_delay - delay);
        if (corrected_delay < tol) {
            std::cout << "img: " << image_time.seconds() << " sys: " << sys_time.seconds()
                      << " cdt: " << corrected_delay.seconds() << " ad: " << actual_delay.seconds();
            count++;
            if (corrected_delay < best_time ) {
                best_time = corrected_delay;
                head.stamp = pair.second.gps_time;
                event_num = pair.second.event_num;
                best_sys_time = sys_time;
                have_best = true;
                std::cout << " *";
            }
            std::cout << std::endl;
        }
    }
    RCLCPP_INFO_STREAM(LOG, "Matched " << count << "/" << event_map_copy.size() << " time headers");
    if (count >= 1) {
        if (remove_when_found && have_best) {
            event_map.erase(best_sys_time);
        }
        return true;
    }
    return false;
}

void EventCache::purge() {
    std::lock_guard<std::mutex> guard(mutex_);
    auto now = clock_.now();
    std::map<rclcpp::Time, custom_msgs::msg::GsofEvt> event_map_copy(event_map);
    for ( const auto &pair : event_map_copy ) {
        auto age = now - rclcpp::Time(pair.second.sys_time);
        if (age > stale_time) {
            event_map.erase(pair.first);
        }
    }
}


std::map<std::string, std::string> parseParams(std::string parameters) {
    // Iterate through parameters organized by name=value, separated by
    // commas. Return a map of parameters to values.
    std::map<std::string, std::string> param_to_value;
    std::string delimiter1 = ",";
    std::string delimiter2 = "=";
    size_t pos = 0;
    std::string token;
    std::string name;
    std::string value;
    // Always run at least once even if there's no delimiter in request
    while ((pos = parameters.find(delimiter1)) != std::string::npos) {
        token = parameters.substr(0, pos);
        name = token.substr(0, token.find(delimiter2));
        token.erase(0, token.find(delimiter2) + delimiter2.length());
        value = token;
        param_to_value[name] = value;
        parameters.erase(0, pos + delimiter1.length());
    }
    token = parameters;
    name = token.substr(0, token.find(delimiter2));
    token.erase(0, token.find(delimiter2) + delimiter2.length());
    value = token;
    param_to_value[name] = value;
    return param_to_value;
};


std::vector<std::string> loadFile(std::string filename) {
    std::vector<std::string> lines;
    std::ifstream inputFile(filename);
        // Check if the file exists and can be opened
    if (!inputFile.is_open()) {
        std::cout << "File " << filename << " does not exist or cannot be opened." << std::endl;
        return lines;
    } else {
        std::string line;
        while (std::getline(inputFile, line)) {
            lines.push_back(line);
        }
    }
    return lines;
};
