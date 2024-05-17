/*********************************************************************
* Software License Agreement (BSD License)
* 
*  Copyright (c) 2008, Willow Garage, Inc.
*  All rights reserved.
* 
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
* 
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
* 
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

#ifndef PROSILICA_H
#define PROSILICA_H

#include <stdexcept>
#include <string>
#include <sstream>
#include <condition_variable>

#include <boost/function.hpp>
#include <boost/thread.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

/// only for defining MetaFrame struct
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

// PvApi.h isn't aware of the usual detection macros
//  these include support for i386, x86_64, and arm, on Linux and OSX
#define _LINUX
#define _x86
#include <prosilica_gige_sdk/PvApi.h>
#undef _LINUX
#undef _x86

void cb_fail_shutdown(ros::TimerEvent e) {
    ROS_WARN("Shutting down due to unhealthy");
    ros::requestShutdown();
}

template <class T>
inline std::string to_string (const T& t)
{
    std::stringstream ss;
    ss << t;
    return ss.str();
}

/// Logging extras
extern uint8_t G_INFO_VERBOSITY;

// ansi color codes
#define BLU "\033[0;34m"
#define GRN "\033[0;32m"
#define RED "\033[0;31m"
#define PUR "\033[0;35m"
// No Color
#define NC "\033[0m"

#define ROS_GREEN(mystr) ROS_INFO(GRN mystr NC)
#define ROS_BLUE(mystr) ROS_INFO(BLU mystr NC)
#define ROS_PUR(mystr) ROS_INFO(PUR mystr NC)

class Watchdog {
public:
    Watchdog();
    Watchdog(double lookback_period, double health_threshold);

    void DelayedStart(ros::NodeHandlePtr nhp, double t);
    void Start();
    void Stop();
    bool Ok();

    void push_back(ros::Time const &t, double val);

    int size();

    void show();

    void purge();

    /// good puppy. positive health event
    void pet();
    /// bad puppy. negative health event
    void kick();
    /// check health status, and trigger any conditions
    void check();

    double computeHealth();

    void setFailCallback(ros::TimerCallback callback);
    void callFail();


private:
    bool enabled_{false};

    /// lookback period in seconds  to consider events
    ros::Duration lookback_period_{30};

    /// 0 = balanced 50/50
    double health_threshold{0.0};
    std::mutex mutex_;
    std::vector<std::pair<ros::Time, double>> array;
    ros::TimerCallback failCallback_;
    ros::Timer delayStartTimer_;
};

namespace prosilica {

ros::Time CvtPvTimestamp(uint32_t timehi, uint32_t timelo);
ros::Time CvtPvTimestamp(uint32_t timehi, uint32_t timelo, uint32_t freq);


struct ProsilicaException : public std::runtime_error
{
  tPvErr error_code;
  
  ProsilicaException(tPvErr code, const char* msg)
    : std::runtime_error(msg), error_code(code)
  {}
};
struct CameraInfo
{
    std::string serial;
    std::string name;
    std::string guid;
    std::string ip_address;
    bool access;
};

void init();                  // initializes API
void fini();                  // releases internal resources
size_t numCameras();          // number of cameras found
uint64_t getGuid(size_t i);   // camera ids
std::vector<CameraInfo> listCameras(); // get list of cameras available
std::string getIPAddress(uint64_t guid); //get ip address data from guid

/**
 * This may need its endianness flipped. Genicam Chunk Mode is weird
 * Not yet implemented.
 *
 */
typedef struct {
    uint32_t acquisitionCount;  // [Bytes 1 - 4]
    uint32_t userValue;         // [Bytes 5 - 8] Not currently implemented. 0.
    uint32_t exposureValue;     // [Bytes 9 - 12] exposure in microseconds
    uint32_t gainValue;         // [Bytes 13 - 16]gain value in units
    uint16_t syncInLevels;      // [Bytes 17 - 18] no idea
    uint16_t syncOutLevels;     // [Bytes 19 - 20] no idea
    char _notImplemented[20];   // [Bytes 21 - 40] not implemented. 0.
    uint32_t chunkId;           // [Bytes 41 - 44] chunk ID. 1000.
    uint32_t chunkLen;          // [Bytes 45 - 48] chunk length.

} AncillaryBuffer;

/**
 * An RAII version of tPvApi Frame buffer.
 */
class PvFrameWrapper {
public:
    PvFrameWrapper(tPvFrame *frame) {
        ROS_BLUE("Copying buffer");
        frame_.AncillaryBufferSize  = frame->AncillaryBufferSize;
        frame_.ImageBufferSize  = frame->ImageBufferSize;
        frame_.ImageSize        = frame->ImageSize;
        frame_.Status           = frame->Status;
        frame_.Width            = frame->Width;
        frame_.Height           = frame->Height;
        frame_.RegionX          = frame->RegionX;
        frame_.RegionY          = frame->RegionY;
        frame_.Format           = frame->Format;
        frame_.BitDepth         = frame->BitDepth;
        frame_.BayerPattern     = frame->BayerPattern;
        frame_.TimestampLo      = frame->TimestampLo;
        frame_.TimestampHi      = frame->TimestampHi;
        frame_.FrameCount       = frame->FrameCount;
        frame_.ImageBuffer = malloc(frame->ImageBufferSize);
        frame_.AncillaryBuffer = malloc(frame->AncillaryBufferSize);
        if (!frame_.ImageBuffer) { throw std::overflow_error("malloc failed on ImageBuffer");}
        if (!frame_.AncillaryBuffer) { throw std::overflow_error("malloc failed on AncillaryBuffer");}
        std::memcpy(frame_.ImageBuffer, frame->ImageBuffer, frame->ImageBufferSize);
        std::memcpy(frame_.AncillaryBuffer, frame->AncillaryBuffer, frame->AncillaryBufferSize);
        std::memcpy(frame_.Context, frame->Context, sizeof(frame->Context));
        std::memcpy(frame_._reserved1, frame->_reserved1, sizeof(frame->_reserved1));
        std::memcpy(frame_._reserved2, frame->_reserved2, sizeof(frame->_reserved2));
    }

    static std::shared_ptr<PvFrameWrapper> make_shared(tPvFrame *frame) {
        return std::make_shared<PvFrameWrapper>(frame);
    }

    ~PvFrameWrapper() {
        free(frame_.ImageBuffer);
        free(frame_.AncillaryBuffer);
    }
    tPvFrame frame_;
};

typedef std::shared_ptr<PvFrameWrapper> PvFrameWrapperPtr;

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


/// According to FrameStartTriggerMode Enum - AVT GigE Camera and Driver Attributes
/// Firmware 1.38 April 7,2010
enum FrameStartTriggerMode
{
  Freerun, 
  SyncIn1, 
  SyncIn2, 
  FixedRate, 
  Software,
  None
};

enum AcquisitionMode
{
  Continuous,
  SingleFrame,
  MultiFrame,
  Recorder
};

enum AutoSetting
{
  Manual,
  Auto,
  AutoOnce
};

/// forward
class Camera;
class MetaFrame;

/** Ok, this is some WEIRD STUFF, so let me explain what is going on.
 The PvApi provides a (void*) Context field, for associating some meta/data
 with a given frame buffer. This used to be done by a raw pointer cast. This
 worked fine with the original driver, however it gave no way to manage
 multiple callback queues easily, since it only returned the same Camera* for
 every frame. When I tried using 4 MetaFrame objects instead, there was a rare
 race condition which could corrupt Context, try to deref and segfault. This
 construction below is kind of like a "custom heap" where instead of dereferencing
 raw pointers cast from void*, it looks up the structure in a map, and throws
 a more useful error when it fails, making it easier to debug where the failure
 is occurring.

    The signature from PvAPI for Context is :
 void*               Context[4]; // 32 bytes on x64

 So actually you get 4 pointers to utilize.
 */
namespace Globals {
    extern std::map<uint32_t, Camera*> camera_registry_;
    extern std::map<void*, std::shared_ptr<MetaFrame>> meta_frame_registry_2;

    static bool register_camera(uint32_t i, Camera *camera);
    static bool register_meta_frame(void* i, std::shared_ptr<MetaFrame> meta_frame);

    static Camera *get_camera(uint32_t i);
    static std::shared_ptr<MetaFrame> get_meta_frame(void* i);


}



/// The original driver uses tPvFrame.Context to point to the Camera struct. However this means there is no way to
/// associate any additional metadata to a given frame. This adds a level of indirection so one can still
/// recover the Camera pointer but we can point the Context to this object and add additional
/// per-frame fields as necessary
class MetaFrame {
public:
    MetaFrame();
    void bind(Camera *camera_, tPvFrame *frame_, unsigned int idx_);

    Camera *camera; // points to camera associated with tPvFrame
    tPvFrame *frame; // points to the actual frame
    unsigned int idx = 0;

    /// Allow for simultaneous post-processing by giving each buffer its own mutex and image message
    /// This violates some of the isolation of the non-ros and ros code, but this driver is already
    /// heavily modified. :shrug:. The trailing underscore matches the prosilica_nodelet convention.
    boost::mutex frameMutex_;
    sensor_msgs::Image img_;
    sensor_msgs::Image broken;
    ros::Timer postProcTimer; // for doing things like dumping image

};

class Camera
{
public:
  static const size_t DEFAULT_BUFFER_SIZE = 4;
  
  Camera(unsigned long guid, size_t bufferSize = DEFAULT_BUFFER_SIZE);
  Camera(const char* ip_address, size_t bufferSize = DEFAULT_BUFFER_SIZE);

  ~Camera();

  //! Must be used before calling start() in a non-triggered mode.
  void setFrameCallback(boost::function<void (tPvFrame*)> callback);
  void setFrameRate(tPvFloat32 frame_rate);

  void setKillCallback(boost::function<void (unsigned long)> callback);
  //! Start capture.
  void start(FrameStartTriggerMode = Freerun, tPvFloat32 frame_rate = 30, AcquisitionMode = Continuous);
  //! Stop capture.
  void stop();
  //! remove callback
  void removeEvents();
  //! Capture a single frame from the camera. Must be called after
  //! start(Software Triggered).
  tPvFrame* grab(unsigned long timeout_ms = PVINFINITE);

  void setExposure(unsigned int val, AutoSetting isauto = Manual);
  void setGain(unsigned int val, AutoSetting isauto = Manual);
  void setWhiteBalance(unsigned int blue, unsigned int red,
                       AutoSetting isauto = Manual);

  void setRoi(unsigned int x, unsigned int y,
              unsigned int width, unsigned int height);
  void setRoiToWholeFrame();
  void setBinning(unsigned int binning_x = 1, unsigned int binning_y = 1);

  //! Returns true if camera supports the attribute.
  bool hasAttribute(const std::string &name);

  //! General get/set attribute functions.
  void getExposure(const std::string &name, tPvUint32 &value, tPvErr &err);
  void getAttribute(const std::string &name, tPvUint32 &value, tPvErr &err);
  void getAttributeEnum(const std::string &name, std::string &value);
  void getAttribute(const std::string &name, tPvUint32 &value);
  void getAttribute(const std::string &name, tPvFloat32 &value);
  void getAttribute(const std::string &name, std::string &value);

  void getAnyAttribute(const std::string &name, std::string &value, std::string &err);

  
  void setAttributeEnum(const std::string &name, const std::string &value);
  void setAttribute(const std::string &name, tPvUint32 value);
  void setAttribute(const std::string &name, tPvFloat32 value);
  void setAttribute(const std::string &name, const std::string &value);

  void runCommand(const std::string& name);

  void dumpAttributeList();
  
  unsigned long guid();

  unsigned long getMaxDataRate();
  static const unsigned long GIGE_MAX_DATA_RATE = 115000000;
  unsigned long max_data_rate= GIGE_MAX_DATA_RATE;
  
  //! Data must have size <= USER_MEMORY_SIZE bytes.
  static const size_t USER_MEMORY_SIZE = 512;
  void writeUserMemory(const char* data, size_t size);
  void readUserMemory(char* data, size_t size);

  //! Get raw PvApi camera handle.
  tPvHandle handle();
  
private:
  std::vector<std::shared_ptr<MetaFrame>> meta_frames_;
  tPvHandle handle_; // handle to open camera
  tPvFrame* frames_; // array of frame buffers
  std::vector<std::shared_ptr<AncillaryBuffer>> anc_buffers_; // array of ancillary buffers
  tPvUint32 frameSize_; // bytes per frame
  size_t bufferSize_; // number of frame buffers
  FrameStartTriggerMode FSTmode_;
  AcquisitionMode Amode_;
  boost::function<void (tPvFrame*)> userCallback_;
  boost::function<void (unsigned long UniqueId)> killCallback_;
  boost::mutex frameMutex_;
  boost::mutex aliveMutex_;
  size_t bufferIndex_;

  void setup();

  static void cameraEventCallback(void* Context,
                                  tPvHandle Camera,
                                  const tPvCameraEvent* EventList,
                                  unsigned long EventListLength);
  static void frameDone(tPvFrame* frame);
  static void kill(void* Context,
                    tPvInterface Interface,
                    tPvLinkEvent Event,
                    unsigned long UniqueId);
};

tPvErr getAttribute(tPvHandle handle, const std::string &name, tPvUint32 &value);
tPvErr getAttribute(tPvHandle handle, const std::string &name, std::string &value);
tPvErr getAttributeList(tPvHandle handle, std::vector<const char*> &infoList);
void dumpAttributeList(tPvHandle handle);

} // namespace prosilica

#endif

