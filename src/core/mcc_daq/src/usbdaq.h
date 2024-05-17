#ifndef MCC_DAQ_USBDAQ_H
#define MCC_DAQ_USBDAQ_H

#include <cstdio>
#include <ros/ros.h>
#include "std_msgs/String.h"
#include "std_msgs/UInt32.h"

#include "usb-2408.h"

#define HIGH 0
#define LOW 1

enum MccDeviceType {MCC_UNDEF=0, MCC_USB2408, MCC_USB2408_AO};

class UsbDaqBase {
public:
    virtual int init() = 0;
    virtual void buildGainTables() = 0;
    virtual const void blip() = 0;
    virtual void pulse(const std_msgs::UInt32::ConstPtr& msg) = 0;
    virtual void switchboard(const std_msgs::String::ConstPtr& msg) = 0;
    virtual void set_blip_micros(const std_msgs::UInt32::ConstPtr &msg) = 0;
    virtual void set_blip_micros(uint32_t period) = 0;
    virtual int analogRead( uint8_t channel) = 0;
    virtual int analogRead( uint8_t channel, uint8_t mode, uint8_t range, uint8_t rate) = 0;
    virtual double voltageRead( uint8_t channel, uint8_t range) = 0;
    virtual void digitalWrite(uint8_t pin, uint8_t value) = 0;
    virtual void digitalPulse(uint8_t pin, uint32_t time_us) = 0;
    virtual void digitalPulse() = 0;

};


class UsbDaqDummy{
//class UsbDaqDummy: public UsbDaqBase {
private:
    libusb_device_handle *udev = nullptr;
    ros::NodeHandlePtr node;
//    ros::NodeHandle nh_dummy();
    MccDeviceType device_type = MCC_UNDEF;
    // I have no idea what I'm doing with these silly handles.
    ros::Publisher     analog_pub ;//=  nh_dummy.advertise<std_msgs::String> ("qux", 1); //<std_msgs::String>
    ros::Publisher     bus_pub ;

    void (*routine)();
    uint8_t outpin = 0;
    uint8_t inpin = 4;
    uint8_t flags = 0;
    uint32_t blip_on_micros = 50000; // on-time of blip function, microseconds

    double gain_table_AIN[NGAINS_2408][2];

public:
    UsbDaqDummy(void);
    explicit UsbDaqDummy(ros::NodeHandlePtr &nh);
    int init();
    void buildGainTables();

    const void blip();
    void pulse(const std_msgs::UInt32::ConstPtr& msg);
    void switchboard(const std_msgs::String::ConstPtr& msg);

    void set_blip_micros(const std_msgs::UInt32::ConstPtr &msg);
    void set_blip_micros(uint32_t period);


    int analogRead( uint8_t channel);
    int analogRead( uint8_t channel, uint8_t mode, uint8_t range, uint8_t rate);
    double voltageRead( uint8_t channel, uint8_t range);
    void digitalWrite(uint8_t pin, uint8_t value);
    void digitalPulse(uint8_t pin, uint32_t time_us);
    void digitalPulse();

};


class UsbDaq{
//class UsbDaq: public UsbDaqBase {
private:
    libusb_device_handle *udev = nullptr;
    ros::NodeHandlePtr node;
//    ros::NodeHandle nh_dummy();
    MccDeviceType device_type = MCC_UNDEF;
    // I have no idea what I'm doing with these silly handles.
    ros::Publisher     analog_pub ;//=  nh_dummy.advertise<std_msgs::String> ("qux", 1); //<std_msgs::String>
    ros::Publisher     bus_pub ;

    void (*routine)();
    uint8_t outpin = 0;
    uint8_t inpin = 4;
    uint8_t flags = 0;
    uint32_t blip_on_micros = 50000; // on-time of blip function, microseconds

    double gain_table_AIN[NGAINS_2408][2];

public:
    UsbDaq(void);
    explicit UsbDaq(ros::NodeHandlePtr &nh);
    int init();
    void buildGainTables();

    void blink(uint8_t count=3);
    const void blip();
    void pulse(const std_msgs::UInt32::ConstPtr& msg);
    void switchboard(const std_msgs::String::ConstPtr& msg);

    void set_blip_micros(const std_msgs::UInt32::ConstPtr &msg);
    void set_blip_micros(uint32_t period);

    void call();

    int analogRead( uint8_t channel);
    int analogRead( uint8_t channel, uint8_t mode, uint8_t range, uint8_t rate);
    int analogWrite( uint8_t channel, double voltage, double table_AO[NCHAN_AO_2408][2]);
    double voltageRead( uint8_t channel, uint8_t gain);
    void digitalWrite(uint8_t pin, uint8_t value);
    void digitalPulse(uint8_t pin, uint32_t time_us);
    void digitalPulse();
    void bind_callback(void (*vfn)());

    void bind_publisher(ros::NodeHandle &nh);
};




/** Exception for initialization unable to find an appropriate USB device
 */
class DeviceNotFoundError
        : public std::exception {
public:
    DeviceNotFoundError() = default;

    virtual ~DeviceNotFoundError() = default;

    virtual char const *what() const noexcept {
        return "No appropriate USB device found";
    }
};

/** Exception for when lacking the right API calls to a specific make of an MCC device.
 * This is mostly here to prevent leaving confusing dead ends for future developers.
 */
class DeviceSpecError
        : public std::exception {
public:
    DeviceSpecError() = default;

    virtual ~DeviceSpecError() = default;

    virtual char const *what() const noexcept {
        return "Could not interface with the appropriate device. Could be an incorrect manufacturer ID, "
               "missing method, or some other mismatch with the device-specific bindings.";
    }
};


/** Failed to initialize libUSB during init
 */
class LibUSBError
        : public std::exception {
public:
    LibUSBError() = default;

    virtual ~LibUSBError() = default;

    virtual char const *what() const noexcept {
        return "Failed to initialize libusb";
    }
};

#endif //MCC_DAQ_USBDAQ_H
