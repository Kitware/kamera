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

#include "prosilica/prosilica.h"
#include <prosilica_gige_sdk/PvRegIo.h>
#include <cassert>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <arpa/inet.h>

#include <ros/console.h>
#include <nlohmann/json.hpp>
#include <sw/redis++/redis++.h>
using namespace sw;

#define CHECK_ERR(fnc, amsg)                               \
do {                                                       \
  tPvErr err = fnc;                                        \
  if (err != ePvErrSuccess) {                              \
    char msg[256];                                         \
    snprintf(msg, 256, "%s: %s", amsg, errorStrings[err]); \
    throw ProsilicaException(err, msg);                    \
  }                                                        \
} while (false)


Watchdog::Watchdog() : failCallback_{cb_fail_shutdown} {}
Watchdog::Watchdog(double lookback_period, double health_threshold)
        :
        lookback_period_{lookback_period},
        health_threshold{health_threshold},
        failCallback_{cb_fail_shutdown} {}

/**Start the watchdog. Events are ignored until started.
 *
 */
void Watchdog::Start() {
    ROS_INFO("watchdog started");
    this->enabled_ = true;
}

/** Start after some delay
 *
 * @param t - delay (seconds)
 */
void Watchdog::DelayedStart(ros::NodeHandlePtr nhp, double t) {
    delayStartTimer_ = nhp->createTimer(ros::Duration{t}, [this](ros::TimerEvent const &e ) {
        this->Start();
    }, true, true);
}

void Watchdog::Stop() {
    this->enabled_ = false;
}

bool Watchdog::Ok() {
    return this->computeHealth() > health_threshold;
}

void Watchdog::push_back(ros::Time const &t, double val) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!t.isValid()) {
        ROS_ERROR("zero/invalid time encountered in Watchdog::push_back()");
        return;
    }
    array.emplace_back(std::pair<ros::Time, double>(t, val));
}

int Watchdog::size() {
    std::lock_guard<std::mutex> guard(mutex_);
    return (int) (array.size());
}

void Watchdog::show() {
    std::lock_guard<std::mutex> guard(mutex_);
    for (const auto pair : array) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
    std::cout << "\n---" << std::endl;
}

void Watchdog::purge() {
    std::lock_guard<std::mutex> guard(mutex_);
    auto now = ros::Time::now();
    std::size_t index = 0;
    for (const auto pair : array) {
        auto age = now - pair.first ;
        if (age > lookback_period_) {
            auto it = array.begin() + index;
            array.erase(it);
        }
        index++;
    }
}

void Watchdog::pet() {
    ROS_INFO("Pet the watchdog");
    this->push_back(ros::Time::now(), 1.0);
}

void Watchdog::kick() {
    if (!this->enabled_) {
        return;
    }
    ROS_WARN("kicked the watchdog");
    this->push_back(ros::Time::now(), -1.0);
}

void Watchdog::check() {
    this->purge();
    if (!this->enabled_) {
        return;
    }
    double health = computeHealth();
    ROS_INFO("current health: %3.3f", health);
    if (health < health_threshold) {
        this->callFail();
    }
}

double Watchdog::computeHealth() {
    double total(array.size());
    double accu = 0;
    for (const auto pair : array) {
        accu += pair.second;
    }
    return accu / total;
}

void Watchdog::setFailCallback(ros::TimerCallback callback) {
    failCallback_ = callback;
}

void Watchdog::callFail() {
    ROS_ERROR("failed health check");
    ros::TimerEvent e;
    failCallback_(e);
}

namespace prosilica {

    namespace Globals {

        std::map<uint32_t, Camera*> camera_registry_;
        std::map<void*, std::shared_ptr<MetaFrame> > meta_frame_registry_;

        bool register_camera(uint32_t i, Camera *camera) {
        camera_registry_[i] = camera;
        }
        bool register_meta_frame(void* i, std::shared_ptr<MetaFrame> meta_frame) {
        meta_frame_registry_[i] = meta_frame;
        }

        Camera *get_camera(uint32_t i) {
        auto el = camera_registry_.find(i);
        if (el == camera_registry_.end()) {
            throw std::invalid_argument("Could not retrieve camera pointer. This usually means a data race");
        }
        return camera_registry_[i];
        }
        std::shared_ptr<MetaFrame> get_meta_frame(void* i) {
        auto el = meta_frame_registry_.find(i);
        if (el == meta_frame_registry_.end()) {
            throw std::invalid_argument("Could not retrieve MetaFrame pointer. This usually means a data race");
        }
        return meta_frame_registry_[i];
        }
    }

    ros::Time CvtPvTimestamp(uint32_t timehi, uint32_t timelo) {
        return CvtPvTimestamp(timehi, timelo, 1000000000);
    }
    ros::Time CvtPvTimestamp(uint32_t timehi, uint32_t timelo, uint32_t freq) {
        uint64_t utime = ((uint64_t )timehi) << 32;
        utime = utime + (uint64_t )timelo;
        double dtime = double (utime) / (double) freq;
        if (dtime <= 0) {
            ROS_ERROR("Calculated time is non-positive: %lf", dtime);
            return ros::Time{};
        }
        return ros::Time{dtime};
    }


static const unsigned int MAX_CAMERA_LIST = 10;
static const char* autoValues[] = {"Manual", "Auto", "AutoOnce"};
static const char* triggerModes[] = {"Freerun", "SyncIn1", "SyncIn2", "FixedRate", "Software"};
static const char* acquisitionModes[] = {"Continuous","SingleFrame","MultiFrame","Recorder"};
static const char* errorStrings[] = {"No error",
                                     "Unexpected camera fault",
                                     "Unexpected fault in PvApi or driver",
                                     "Camera handle is invalid",
                                     "Bad parameter to API call",
                                     "Sequence of API calls is incorrect",
                                     "Camera or attribute not found",
                                     "Camera cannot be opened in the specified mode",
                                     "Camera was unplugged",
                                     "Setup is invalid (an attribute is invalid)",
                                     "System/network resources or memory not available",
                                     "1394 bandwidth not available",
                                     "Too many frames on queue",
                                     "Frame buffer is too small",
                                     "Frame cancelled by user",
                                     "The data for the frame was lost",
                                     "Some data in the frame is missing",
                                     "Timeout during wait",
                                     "Attribute value is out of the expected range",
                                     "Attribute is not this type (wrong access function)",
                                     "Attribute write forbidden at this time",
                                     "Attribute is not available at this time",
                                     "A firewall is blocking the traffic"};

static const char* pvDataTypeStrings[] = {
        "Unknown",
        "Command",
        "Raw",
        "String",
        "Enum",
        "Uint32",
        "Float32",
        "Int64",
        "Boolean",
};

static tPvCameraInfo cameraList[MAX_CAMERA_LIST];
static unsigned long cameraNum = 0;

std::string fmt_attribute_info(tPvAttributeInfo *info) {
    std::stringstream out;
    out << pvDataTypeStrings[info->Datatype];
    return out.str();
}

std::string makeAttrError(std::string const &msg, std::string const &name) {
    return "{\"msg\": \"" + msg + "\", \"name\": \"" + name + "\"}";
}
std::string makeAttrWrapper(std::string const &name, std::string const &value, std::string const &dtype) {
    return "{\"name\": \"" + name + "\", \"value\": \"" + value + "\", \"dtype\": \"" + dtype + "\"}";
}

MetaFrame::MetaFrame() {};

void MetaFrame::bind(Camera *camera_, tPvFrame *frame_, unsigned int idx_) {
    camera = camera_;
    frame = frame_;
    idx = idx_;
}


void init()
{
  CHECK_ERR( PvInitialize(), "Failed to initialize Prosilica API" );
}

void fini()
{
  PvUnInitialize();
}

size_t numCameras()
{
  cameraNum = PvCameraList(cameraList, MAX_CAMERA_LIST, NULL);
  return cameraNum;
}

uint64_t getGuid(size_t i)
{
  assert(i < MAX_CAMERA_LIST);
  if (i >= cameraNum)
    throw ProsilicaException(ePvErrBadParameter, "No camera at index i");
  return cameraList[i].UniqueId;
}

std::vector<CameraInfo> listCameras()
{
    std::vector<CameraInfo> cameras;
    tPvCameraInfo cameraList[MAX_CAMERA_LIST];
    unsigned long cameraNum = 0;
    //! get list of all cameras
    cameraNum = PvCameraList(cameraList, MAX_CAMERA_LIST, NULL);
    //! append list of unreachable cameras
    if (cameraNum < MAX_CAMERA_LIST)
    {
        cameraNum += PvCameraListUnreachable(&cameraList[cameraNum], MAX_CAMERA_LIST-cameraNum, NULL);
    }
    if(cameraNum)
    {
        struct in_addr addr;
        tPvIpSettings Conf;

        //! get info
        for(unsigned long i=0; i < cameraNum; i++)
        {

            CameraInfo camInfo;
            camInfo.serial     = to_string(cameraList[i].SerialString);
            camInfo.name       = to_string(cameraList[i].DisplayName);
            camInfo.guid       = to_string(cameraList[i].UniqueId);
            PvCameraIpSettingsGet(cameraList[i].UniqueId,&Conf);
            addr.s_addr = Conf.CurrentIpAddress;
            camInfo.ip_address = to_string(inet_ntoa(addr));
            camInfo.access     = cameraList[i].PermittedAccess & ePvAccessMaster ? true : false;

            cameras.push_back(camInfo);
        }
    }
    return cameras;
}

std::string getIPAddress(uint64_t guid)
{
    struct in_addr addr;
    tPvIpSettings Conf;
    CHECK_ERR(PvCameraIpSettingsGet(guid, &Conf), "Unable to retrieve IP address");
    addr.s_addr = Conf.CurrentIpAddress;
    std::stringstream ip;
    ip<<inet_ntoa(addr);
    return ip.str();
}

/// @todo support opening as monitor?
static void openCamera(boost::function<tPvErr (tPvCameraInfo*)> info_fn,
                       boost::function<tPvErr (tPvAccessFlags)> open_fn)
{
  cameraNum = PvCameraList(cameraList, MAX_CAMERA_LIST, NULL);
  tPvCameraInfo info;
  CHECK_ERR( info_fn(&info), "Unable to find requested camera" );
  ROS_INFO("Cam %ld %s: Access: %lx ", info.UniqueId, info.SerialString, info.PermittedAccess);
  if (!(info.PermittedAccess & ePvAccessMaster))
  {

      ROS_ERROR("Unable to open camera as master. ");
      if (info.PermittedAccess & ePvAccessMonitor) {
          ROS_INFO("Driver can access monitor");
      }
  }

  CHECK_ERR( open_fn(ePvAccessMaster), "Unable to open requested camera" );
}


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



Camera::Camera(unsigned long guid, size_t bufferSize)
  : bufferSize_(bufferSize), FSTmode_(None)
{
  openCamera(boost::bind(PvCameraInfo, guid, _1),
             boost::bind(PvCameraOpen, guid, _1, &handle_));
  
  setup();
}

Camera::Camera(const char* ip_address, size_t bufferSize)
  : bufferSize_(bufferSize), FSTmode_(None)
{
  unsigned long addr = inet_addr(ip_address);
  tPvIpSettings settings;
  openCamera(boost::bind(PvCameraInfoByAddr, addr, _1, &settings),
             boost::bind(PvCameraOpenByAddr, addr, _1, &handle_));
  
  setup();
}

void Camera::setup()
{
    Globals::register_camera(0, this); /// we are only allowing one camera
  // adjust packet size according to the current network capacity
  tPvUint32 maxPacketSize = 9000;
  PvCaptureAdjustPacketSize(handle_, maxPacketSize);

  // set data rate to the max
  max_data_rate = getMaxDataRate();
  if (max_data_rate < GIGE_MAX_DATA_RATE) {
    ROS_WARN("Detected max data rate is %lu bytes/s, typical maximum data rate for a "
             "GigE port is %lu bytes/s. Are you using a GigE network card and cable?\n",
             max_data_rate, GIGE_MAX_DATA_RATE);
  }
  setAttribute("StreamBytesPerSecond", max_data_rate);

  // capture whole frame by default
  setBinning();
  setRoiToWholeFrame();
  
  // query for attributes (TODO: more)
  CHECK_ERR( PvAttrUint32Get(handle_, "TotalBytesPerFrame", &frameSize_),
             "Unable to retrieve frame size" );
  
  // allocate frame buffers
  frames_ = new tPvFrame[bufferSize_];
  memset(frames_, 0, sizeof(tPvFrame) * bufferSize_);
    void*               Context[4];
  ROS_INFO("Size of void*: %lu size_t: %lu Context: %lu", sizeof(void*), sizeof(size_t), sizeof(Context));
  for (size_t i = 0; i < bufferSize_; ++i)
  {
      meta_frames_.emplace_back(std::make_shared<MetaFrame>());
      void* idx = (void*)(&(*meta_frames_[i]));
      Globals::register_meta_frame(idx, meta_frames_[i]);
    meta_frames_[i]->bind(this, &frames_[i], i);
    frames_[i].ImageBuffer = new char[frameSize_];
    frames_[i].ImageBufferSize = frameSize_;
    frames_[i].Context[0] = idx; // for frameDone callback
      ROS_INFO("[2] Raw value: %p", frames_[i].Context[0]);
//      ROS_INFO("[2] Raw value: %x %x %x %x", frames_[i].Context[0], frames_[i].Context[1], frames_[i].Context[2], frames_[i].Context[3]  );

  }

  PvLinkCallbackRegister(Camera::kill, ePvLinkRemove, this);
}

Camera::~Camera()
{
  PvLinkCallbackUnRegister(Camera::kill, ePvLinkRemove);
  stop();
  
  PvCameraClose(handle_);

  if (frames_) {
    for (unsigned int i = 0; i < bufferSize_; ++i)
      delete[] (char*)frames_[i].ImageBuffer;
    delete[] frames_;
  }
  meta_frames_.clear();
  anc_buffers_.clear();
}

void Camera::setFrameCallback(boost::function<void (tPvFrame*)> callback)
{
  userCallback_ = callback;
}
void Camera::setFrameRate(tPvFloat32 frame_rate){
  CHECK_ERR( PvAttrFloat32Set(handle_, "FrameRate", frame_rate),
	     "Could not set frame rate");
}

void Camera::setKillCallback(boost::function<void (unsigned long UniqueId)> callback)
{
    killCallback_ = callback;
}

void Camera::start(FrameStartTriggerMode fmode, tPvFloat32 frame_rate, AcquisitionMode amode)
{
    assert( FSTmode_ == None && fmode != None );
    ///@todo verify this assert again
    assert( fmode == SyncIn1 || fmode == SyncIn2 || fmode == Software || fmode == FixedRate || !userCallback_.empty() );

    // set camera in acquisition mode
    CHECK_ERR( PvCaptureStart(handle_), "Could not start capture");

    if (fmode == Freerun || fmode == FixedRate || fmode == SyncIn1 || fmode == SyncIn2)
    {
        for (unsigned int i = 0; i < bufferSize_; ++i) {
            // main frame reading loop
            PvCaptureQueueFrame(handle_, frames_ + i, Camera::frameDone);
        }
        PvCameraEventCallbackRegister(handle_, Camera::cameraEventCallback, nullptr);

    }
    else
    {
        bufferIndex_ = 0;
        CHECK_ERR( PvCaptureQueueFrame(handle_, &frames_[bufferIndex_], NULL), "Could not queue frame");
    }

    // start capture after setting acquisition and trigger modes
    try {
        ///@todo take this one also as an argument
        CHECK_ERR( PvAttrEnumSet(handle_, "AcquisitionMode", acquisitionModes[amode]),
                   "Could not set acquisition mode" );
        CHECK_ERR( PvAttrEnumSet(handle_, "FrameStartTriggerMode", triggerModes[fmode]),
                   "Could not set trigger mode" );
        CHECK_ERR( PvCommandRun(handle_, "AcquisitionStart"),
                   "Could not start acquisition" );
    }
    catch (ProsilicaException& e) {
        stop();
        throw; // rethrow
    }
    FSTmode_ = fmode;
    Amode_ = amode;
    
    CHECK_ERR( PvAttrFloat32Set(handle_, "FrameRate", frame_rate),
	           "Could not set frame rate");
    ROS_INFO("End of Camera::start()");
}

void Camera::stop()
{
  if (FSTmode_ == None)
    return;
  ROS_WARN("Stopping camera");
  PvCommandRun(handle_, "AcquisitionStop");
  PvCaptureQueueClear(handle_);
  PvCaptureEnd(handle_);
  FSTmode_ = None;
ROS_INFO("End of Camera::stop()");

}

void Camera::removeEvents()
{
    // clear all events
    PvAttrUint32Set(handle_, "EventsEnable1", 0);
}

tPvFrame* Camera::grab(unsigned long timeout_ms)
{
    assert( FSTmode_ == Software );
    tPvFrame* frame = &frames_[0];

    CHECK_ERR( PvCommandRun(handle_, "FrameStartTriggerSoftware"), "Couldn't trigger capture" );
    CHECK_ERR( PvCaptureWaitForFrameDone(handle_, frame, timeout_ms), "couldn't capture frame");
    // don't requeue if capture has stopped
    if (frame->Status == ePvErrUnplugged || frame->Status == ePvErrCancelled )
    {
      return NULL;
    }
    CHECK_ERR( PvCaptureQueueFrame(handle_, frame, NULL), "Couldn't queue frame");

    if (frame->Status == ePvErrSuccess)
        return frame;
    if (frame->Status == ePvErrDataMissing || frame->Status == ePvErrTimeout)
    {
        //! recommanding after an error seems to cause a sequence error if next command is too fast
        boost::this_thread::sleep(boost::posix_time::millisec(50));
        return NULL;
    }
    else
        throw std::runtime_error("Unknown error grabbing frame");

    return frame;
}

void Camera::setExposure(unsigned int val, AutoSetting isauto)
{
    printf("\n<> SET EXPOSURE = %d \n", val);
  CHECK_ERR( PvAttrEnumSet(handle_, "ExposureMode", autoValues[isauto]),
             "Couldn't set exposure mode" );

  if (isauto == Manual)
    CHECK_ERR( PvAttrUint32Set(handle_, "ExposureValue", val),
               "Couldn't set exposure value" );
}

void Camera::setGain(unsigned int val, AutoSetting isauto)
{
  /// @todo Here and in setWhiteBalance, would be better to split off setGainMode etc.
  /// I didn't take into account there are cameras that don't support auto gain, auto white balance.
  if (PvAttrIsAvailable(handle_, "GainMode") == ePvErrSuccess)
  {
    CHECK_ERR( PvAttrEnumSet(handle_, "GainMode", autoValues[isauto]),
               "Couldn't set gain mode" );
  }

  if (isauto == Manual)
    CHECK_ERR( PvAttrUint32Set(handle_, "GainValue", val),
               "Couldn't set gain value" );
}

void Camera::setWhiteBalance(unsigned int blue, unsigned int red, AutoSetting isauto)
{
  if (PvAttrIsAvailable(handle_, "WhitebalMode") == ePvErrSuccess)
  {
    CHECK_ERR( PvAttrEnumSet(handle_, "WhitebalMode", autoValues[isauto]),
               "Couldn't set white balance mode" );
  }

  if (isauto == Manual)
  {
      if(hasAttribute("WhitebalValueBlue"))
      {
        CHECK_ERR( PvAttrUint32Set(handle_, "WhitebalValueBlue", blue),
                   "Couldn't set white balance blue value" );
      }
      if(hasAttribute("WhitebalValueRed"))
      {
        CHECK_ERR( PvAttrUint32Set(handle_, "WhitebalValueRed", red),
                   "Couldn't set white balance red value" );
      }
  }
}

void Camera::setRoi(unsigned int x, unsigned int y,
                    unsigned int width, unsigned int height)
{
  CHECK_ERR( PvAttrUint32Set(handle_, "RegionX", x),
             "Couldn't set region x (left edge)" );
  CHECK_ERR( PvAttrUint32Set(handle_, "RegionY", y),
             "Couldn't set region y (top edge)" );
  CHECK_ERR( PvAttrUint32Set(handle_, "Width", width),
             "Couldn't set region width" );
  CHECK_ERR( PvAttrUint32Set(handle_, "Height", height),
             "Couldn't set region height" );
}

void Camera::setRoiToWholeFrame()
{
  tPvUint32 min_val, max_val;
  CHECK_ERR( PvAttrUint32Set(handle_, "RegionX", 0),
             "Couldn't set region x (left edge)" );
  CHECK_ERR( PvAttrUint32Set(handle_, "RegionY", 0),
             "Couldn't set region y (top edge)" );
  CHECK_ERR( PvAttrRangeUint32(handle_, "Width", &min_val, &max_val),
             "Couldn't get range of Width attribute" );
  CHECK_ERR( PvAttrUint32Set(handle_, "Width", max_val),
             "Couldn't set region width" );
  CHECK_ERR( PvAttrRangeUint32(handle_, "Height", &min_val, &max_val),
             "Couldn't get range of Height attribute" );
  CHECK_ERR( PvAttrUint32Set(handle_, "Height", max_val),
             "Couldn't set region height" );
}

void Camera::setBinning(unsigned int binning_x, unsigned int binning_y)
{
  // Permit setting to "no binning" on cameras without binning support
  if (!hasAttribute("BinningX") && binning_x == 1 && binning_y == 1)
    return;
  
  CHECK_ERR( PvAttrUint32Set(handle_, "BinningX", binning_x),
             "Couldn't set horizontal binning" );
  CHECK_ERR( PvAttrUint32Set(handle_, "BinningY", binning_y),
             "Couldn't set vertical binning" );
}

bool Camera::hasAttribute(const std::string &name)
{
  return (PvAttrIsAvailable(handle_, name.c_str()) == ePvErrSuccess);
}

static void getStringValuedAttribute(std::string &value,
  boost::function<tPvErr (char*, unsigned long, unsigned long*)> get_fn)
{
    const size_t sz = 32;
    unsigned long actual_size;
    char tmp[sz];

    CHECK_ERR( get_fn(tmp, sz, &actual_size),
               "Couldn't get attribute" );
    if (actual_size >= sz) {
        size_t new_size = actual_size + 1;
        char *pBuffer = (char *) malloc(new_size);
        get_fn(pBuffer, new_size, &actual_size);
        value = std::string(pBuffer);
        free(pBuffer);
    } else {
        value = std::string(tmp);
    }

}

void Camera::getAttributeEnum(const std::string &name, std::string &value)
{
  getStringValuedAttribute(value,
    boost::bind(PvAttrEnumGet, handle_, name.c_str(), _1, _2, _3));
}

void Camera::getAttribute(const std::string &name, tPvUint32 &value, tPvErr &err)
{
    /** in prog */
    err = PvAttrUint32Get(handle_, name.c_str(), &value);

}

void Camera::getAttribute(const std::string &name, tPvUint32 &value)
{
  std::string err_msg = "Couldn't get attribute " + name;
  CHECK_ERR( PvAttrUint32Get(handle_, name.c_str(), &value),
	     err_msg.c_str());
             
}

void Camera::getAttribute(const std::string &name, tPvFloat32 &value)
{
std::string err_msg = "Couldn't get attribute " + name;
  CHECK_ERR( PvAttrFloat32Get(handle_, name.c_str(), &value),
             err_msg.c_str());
}

void Camera::getAttribute(const std::string &name, std::string &value)
{
  getStringValuedAttribute(value,
    boost::bind(PvAttrStringGet, handle_, name.c_str(), _1, _2, _3));
}

/// WIP
void Camera::getAnyAttribute(const std::string &name, std::string &value, std::string &err)
{
    tPvErr status;
    tPvAttributeInfo pInfo;
    std::string tmp;
    status = PvAttrIsAvailable(handle_, name.c_str());
    if (status != ePvErrSuccess) {
        err = makeAttrError("Attribute not available", name) ;
        return;
    }
    status = PvAttrInfo(handle_, name.c_str(), &pInfo);
    if (status != ePvErrSuccess) {
        err = makeAttrError("Could not get attribute info", name) ;
        return;
    }

    value = makeAttrWrapper(name, tmp, std::string(pvDataTypeStrings[pInfo.Datatype]));

//    getStringValuedAttribute(value,
//    boost::bind(PvAttrStringGet, handle_, name.c_str(), _1, _2, _3));
}

void Camera::setAttributeEnum(const std::string &name, const std::string &value)
{
  std::string err_msg = "Couldn't get attribute " + name;
  CHECK_ERR( PvAttrEnumSet(handle_, name.c_str(), value.c_str()),
             err_msg.c_str());
}

void Camera::setAttribute(const std::string &name, tPvUint32 value)
{
  std::string err_msg = "Couldn't set attribute " + name;
  CHECK_ERR( PvAttrUint32Set(handle_, name.c_str(), value),
             err_msg.c_str());
}

void Camera::setAttribute(const std::string &name, tPvFloat32 value)
{
  std::string err_msg = "Couldn't set attribute " + name;
  CHECK_ERR( PvAttrFloat32Set(handle_, name.c_str(), value),
             err_msg.c_str());
}

void Camera::setAttribute(const std::string &name, const std::string &value)
{
  std::string err_msg = "Couldn't set attribute " + name;
  CHECK_ERR( PvAttrStringSet(handle_, name.c_str(), value.c_str()),
             err_msg.c_str());
}

void Camera::runCommand(const std::string& name)
{
  std::string err_msg = "Couldn't run command " + name;
  CHECK_ERR( PvCommandRun(handle_, name.c_str()), err_msg.c_str());
}

unsigned long Camera::guid()
{
  unsigned long id;
  CHECK_ERR( PvAttrUint32Get(handle_, "UniqueId", &id),
             "Couldn't retrieve unique id" );
  return id;
}

unsigned long Camera::getMaxDataRate()
{
  tPvUint32 min_data_rate, max_data_rate;
  CHECK_ERR( PvAttrRangeUint32(handle_, "StreamBytesPerSecond", &min_data_rate, &max_data_rate),
             "Couldn't get range of attribute StreamBytesPerSecond" );
  return max_data_rate;
}

static const unsigned long USER_ADDRESS = 0x17200;

void Camera::writeUserMemory(const char* data, size_t size)
{
  assert(size <= USER_MEMORY_SIZE);

  unsigned char buffer[USER_MEMORY_SIZE] = {0};
  memcpy(buffer, data, size);

  unsigned long written;
  CHECK_ERR( PvMemoryWrite(handle_, USER_ADDRESS, USER_MEMORY_SIZE, buffer, &written),
             "Couldn't write to user memory" );
}

void Camera::readUserMemory(char* data, size_t size)
{
  assert(size <= USER_MEMORY_SIZE);

  unsigned char buffer[USER_MEMORY_SIZE];
  
  CHECK_ERR( PvMemoryRead(handle_, USER_ADDRESS, USER_MEMORY_SIZE, buffer),
             "Couldn't read from user memory" );

  memcpy(data, buffer, size);
}

/** camera event callback */
void Camera::cameraEventCallback(void* Context, tPvHandle Camera, const tPvCameraEvent* EventList,
                                 unsigned long EventListLength) {
    ROS_WARN("cameraEventCallback");
    for (auto i=0; i < EventListLength; i++) {
        ROS_INFO("callback %lu", EventList[i].EventId);
    }
}

    /** frameDone callback
 * Called at the end of frame transfer off camera.
 * This thing is a bit of a doozy, so I am taking notes.
 * PvCaptureQueueFrame buffs frames from camera to computer. It optionally allows
 * a callback to be provided on success, and returns immediately.
 * This driver uses frameDone to queue up another transfer by recursive calls
 * into frameDone. That means that frameDone only takes milliseconds to run,
 * however it won't fire off until the next frame transfer completes.
 * It also makes it crazy hard to profile.
 * ALSO - logging in this driver can break things ELSEWHERE
 * memcpying the image buffer takes only about 2 ms, so it may be worth to just copy the
 * tPvFrame to a different context and let the pre-processing occur there.
 * sizeof(tPvFrame) = 488, 448 excluding reserved1/2.
 * @param frame
 */
void Camera::frameDone(tPvFrame* frame)
{
//    tic = ros::Time::now();
    if (!frame) {
        throw std::runtime_error("Empty frame, this should never happen");
    }
    ROS_PUR("[0]  === Enter frameDone === ");
        ROS_INFO("[%lu] Raw value: %p status: %d", frame->FrameCount, frame->Context[0] , frame->Status);
        void* idx = (void*)frame->Context[0];
        auto meta_frame = Globals::get_meta_frame(idx);
//    ROS_INFO("[2]Suceess cast frameDone");
    Camera* camPtr = Globals::get_camera(0);
  if (camPtr && !camPtr->userCallback_.empty()) {
//    ROS_INFO("user callback thingy %p #%ld %ld sizeof frame: %ld", meta_frame, (long int) meta_frame->idx, (long int) frame->FrameCount %4, (long int) sizeof(*frame));
    // TODO: thread safety OK here?
    // this mutex ensures there is only 1 thread of userCallback_ at a time
    // I've tried setting a per-buf mutex, but that somehow leads to race conditions
    // that means we are probably stuck with a single mutex guarding this critical region
//    auto start_cb = ros::Time::now();
    boost::lock_guard<boost::mutex> guard(camPtr->frameMutex_);
    camPtr->userCallback_(frame);  // this (usually) routes to .publish!
//    auto end_cb = ros::Time::now();
//    ROS_INFO("Time in cb: %.8f", (end_cb - start_cb).toSec());
  }

//  ROS_INFO("Frame status: %d", frame->Status);

  // don't requeue if capture has stopped
  if (frame->Status == ePvErrUnplugged || frame->Status == ePvErrCancelled)
  {
    return;
  }
    PvCaptureQueueFrame(camPtr->handle_, frame, Camera::frameDone);
//    toc = ros::Time::now();
//    ROS_INFO("Leave frameDone, elapsed: %2.3f", (toc-tic).toSec());
}


void Camera::kill(void* Context,
                   tPvInterface Interface,
                   tPvLinkEvent Event,
                   unsigned long UniqueId)
{
    MetaFrame* meta_frame = (MetaFrame*) Context;
    Camera* camPtr = Globals::get_camera(0);
    if (camPtr && !camPtr->killCallback_.empty())
    {
        //boost::lock_guard<boost::mutex> guard(camPtr->aliveMutex_);
        camPtr->killCallback_(UniqueId);
    }
}

tPvHandle Camera::handle()
{
  return handle_;
}

tPvErr getAttribute(tPvHandle handle, const std::string &name, tPvUint32 &value) {
    return PvAttrUint32Get(handle, name.c_str(), &value);
}


tPvErr getAttributeList(tPvHandle handle, std::vector<const char*> &infoList) {
    const char* const** pListPtr;
    unsigned long pLength;
    tPvErr err = PvAttrList(handle, pListPtr, &pLength);
    if (err != ePvErrSuccess) return err;

    for (auto i = 0; i < pLength; i++ ) {
        auto x = (*pListPtr)[i];
        infoList.emplace_back(x);
    }
    return ePvErrSuccess;
}

void dumpAttributeList(tPvHandle handle) {
    std::vector<const char*> infoList;
    tPvAttributeInfo pInfo;
    getAttributeList(handle, infoList);
    for (auto it = infoList.begin(); it != infoList.end(); ++it) {
        PvAttrInfo(handle, *it, &pInfo);
        std::cout << *it << ": " << pvDataTypeStrings[pInfo.Datatype] <<  "\n";
    }
    std::cout << std::endl;
}

tPvErr getAttribute(tPvHandle handle_, const std::string &name, std::string &value) {
    getStringValuedAttribute(value,
            boost::bind(PvAttrStringGet, handle_, name.c_str(), _1, _2, _3));

}

} // namespace prosilica

