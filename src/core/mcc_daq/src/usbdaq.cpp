#include <stdlib.h>
#include <stdio.h>
#include <chrono>

#include <libusb.h>
#include <rclcpp/rclcpp.hpp>
#include "std_msgs/msg/string.hpp"

#include "pmd.h"
#include "usb-2408.h"
#include "utils.h"
#include "usbdaq.h"


UsbDaq::UsbDaq() {
    // I have no idea why I need this.
}

UsbDaq::UsbDaq(rclcpp::Node::SharedPtr nh) {
    node = nh;
    bus_pub = node->create_publisher<std_msgs::msg::String>("/bus", 10);
    std_msgs::msg::String msg;
    msg.data = "DAQ device going online";
    bus_pub->publish(msg);
}

int UsbDaq::init() {
    int libusb_err = libusb_init(nullptr);
    if (libusb_err < 0) {
        throw LibUSBError();
    }
    std_msgs::msg::String msg;

    if ((udev = usb_device_find_USB_MCC(USB2408_PID, nullptr))) {

        ROS_GREEN("Success, found a USB 2408!");
        msg.data = "Success, found a USB 2408!";
        device_type = MCC_USB2408;
    } else if ((udev = usb_device_find_USB_MCC(USB2408_2AO_PID, nullptr))) {
        ROS_GREEN("Success, found a USB 2408_2AO!\n");
        msg.data = "Success, found a USB 2408_2AO!";
        device_type = MCC_USB2408_AO;
    } else {
        ROS_ERROR("Failure, did not find a USB 2408 or 2408_2AO!\n");
        msg.data = "Failure, did not find a USB 2408 or 2408_2AO!";
        bus_pub->publish(msg);
        throw DeviceNotFoundError();
    }
    bus_pub->publish(msg);
    buildGainTables();
    return 0;
}

/** Builds a lookup table of calibration coefficents to translate values into voltages.
 * see usb-####.c for details
 * */
void UsbDaq::buildGainTables() {
    if (device_type == MCC_USB2408_AO) {
        usbBuildGainTable_USB2408(udev, gain_table_AIN);
    } else {
        throw DeviceSpecError();
    }
}

/** Blink the embedded LED */
void UsbDaq::blink(uint8_t count) {
    usbBlink_USB2408(udev, count);
}

/** Write a value to a specific pin */
void UsbDaq::digitalWrite(uint8_t pin, uint8_t value) {
    usbDOut_USB2408(udev, value, pin);
    ROS_INFO3("Pin: %d Value: %d", pin, value);
}

/** Toggle a pin briefly */
const void UsbDaq::blip() {
    uint8_t pin = 0;
    digitalWrite(pin, HIGH);
    usleep(blip_on_micros);
    digitalWrite(pin, LOW);
}

/** Toggle a pin for time_us microseconds */
void UsbDaq::digitalPulse(uint8_t pin, uint32_t time_us){
    ROS_INFO1("Pulsing for %d us", time_us);
    digitalWrite(pin, HIGH);
    usleep(time_us);
    digitalWrite(pin, LOW);
}

void UsbDaq::digitalPulse() {
    digitalPulse(outpin, 250000);
}

int UsbDaq::analogRead(uint8_t channel) {
    uint8_t range, rate, mode;
    range = BP_10V;
    rate = HZ1000;
    mode = DIFFERENTIAL;
    return analogRead(channel, mode, range, rate);
}

int UsbDaq::analogRead(uint8_t channel, uint8_t mode, uint8_t range, uint8_t rate) {
    int data = usbAIn_USB2408(udev, channel, mode, range, rate, &flags);
    data = data*range;
    return data;
}

int UsbDaq::analogWrite(uint8_t channel, double voltage, double table_AO[NCHAN_AO_2408][2]){
    usbAOut_USB2408_2AO(udev, channel, voltage, table_AO);
    return 0;
}

/**
 *
 * @param channel
 * @param gain - this is gain in test-usb2408.c. voltage range constant
 *  e.g. BP_10V
 * @return
 */
double UsbDaq::voltageRead(uint8_t channel, uint8_t gain) {
    uint8_t rate, mode;
    uint8_t range = BP_10V;
    rate = HZ1000;
    mode = DIFFERENTIAL;
    ROS_INFO("Reading device");

    int data = usbAIn_USB2408(udev, channel, mode, range, rate, &flags);

// I have no idea why the original code does this janky cast
// pretty sure this is y = mx + b
    ROS_INFO("device read");

    double temp = (int) (data * gain_table_AIN[gain][0] + gain_table_AIN[gain][1]);
    double voltage =  volts_USB2408_d(gain, temp);
    return voltage;
}

/** Send out a digital pulse for N microseconds
 *
 * @param msg Uint32 Microseconds to turn on for
 */
void UsbDaq::pulse(const std_msgs::msg::UInt32::ConstSharedPtr &msg) {
    ROS_INFO3("Pulse: [%d]", msg->data);
    digitalPulse(outpin, msg->data);
}

/** Take a string message, parse it, and run a function based off that command
 * hi: set outpin high
 * lo: set outpin low
 * pu###: pulse outpin for ###ms
 * re: read analog on inpin
 * @param msg
 */
void UsbDaq::switchboard(const std_msgs::msg::String::ConstSharedPtr &msg) {
    ROS_INFO3("I heard: [%s]", msg->data.c_str());
    std::string prefix = msg->data.substr(0, 2);
    if (prefix == "hi") {
        digitalWrite(outpin, HIGH);
    } else if (prefix == "lo") {
        digitalWrite(outpin, LOW);
    } else if (prefix == "pu") {
        int time_ms = 0;
        try  {
            time_ms = std::stoi(msg->data.substr(2));
        }
        catch (const std::invalid_argument &) { ROS_WARN("Could not parse that"); }
        if (time_ms) {
            digitalPulse(outpin, time_ms);
        }

    } else if (prefix == "re") {
        uint8_t chan = inpin;
        int data = analogRead(chan);
        std::string volt_str = std::to_string(data);
        ROS_INFO("Voltage: %d [%s]", data, volt_str.c_str());
        ROS_GREEN("Voltage");
    }
}

void UsbDaq::call() {
    ROS_INFO3("call() called");
    routine();
}

/** Bind a function pointer to an internal */
void UsbDaq::bind_callback(void (*vfn)() ) {
    routine = vfn;
}

void UsbDaq::set_blip_micros(uint32_t period) {
    blip_on_micros = period;
}

void UsbDaq::set_blip_micros(const std_msgs::msg::UInt32::ConstSharedPtr &msg) {
    set_blip_micros(msg->data);
}
