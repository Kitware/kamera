#include <iostream>
#include <cstdio>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <thread>
#include <mutex>
#include <algorithm>
#include <boost/filesystem.hpp>

#include <P1Camera.hpp>
#include <P1Image.hpp>
#include <P1ImageJpegWriter.hpp>
#include <P1ImageTiffWriter.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <pluginlib/class_list_macros.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <std_msgs/Header.h>
#include <diagnostic_updater/diagnostic_updater.h>
#include <diagnostic_updater/publisher.h>

#include <roskv/envoy.h>
#include <roskv/archiver.h>
#include <custom_msgs/GSOF_EVT.h>
#include <custom_msgs/Stat.h>
#include <custom_msgs/RequestImageView.h>

#include <phase_one/GetPhaseOneParameter.h>
#include <phase_one/SetPhaseOneParameter.h>
#include <phase_one/GetCompressedImageView.h>
#include <phase_one/GetImageView.h>
#include <phase_one/phase_one_utils.h>
#include <phase_one/phase_one.h>


namespace phase_one
{
    PhaseOne::PhaseOne()
    {
    }

    void PhaseOne::init()
    {
        PhaseOne::onInit();
    }

    PhaseOne::~PhaseOne() {
        ROS_INFO("Phase One Camera: Shutting down");
        // signal running_ threads and wait until they finish
        running_ = false;
        if (capture_thread_.joinable())
        {
            capture_thread_.join();
        }
        if (demosaic_thread_.joinable())
        {
            demosaic_thread_.join();
        }
        // Disable image receiving
        camera.Close();
    };

    void PhaseOne::onInit() {
        ROS_INFO("PhaseOne Driver: Initialization");
        // ROS initialization
        ros::NodeHandle nh;
        ros::NodeHandle pnh("~");
        image_transport::ImageTransport it(nh);
        image_pub = it.advertise("image_raw", 1);
        stat_pub_ = nh.advertise<custom_msgs::Stat>("/stat", 3);

        ROS_INFO("Loading ROS parameters.");
        pnh.param<std::string>("ip_address", ip_address_, "");
        pnh.param<int>("num_threads", num_threads_, 28);
        pnh.param<std::string>("trigger_mode", trigger_mode_, "manual"); //manual, auto
        pnh.param<double>("auto_trigger_rate", auto_trigger_rate_, 1.0);
        pnh.param<std::string>("cam_chan", cam_channel_, "nochan");
        pnh.param<std::string>("cam_fov", cam_fov_, "nofov");
        pnh.param<std::string>("hostname", hostname, "casX");

        // connect to redis
        ROS_INFO("Cameratype: %s/%s", cam_fov_.c_str(), cam_channel_.c_str());
        RedisEnvoyOpts envoy_opts = RedisEnvoyOpts::from_env("driver_" + cam_fov_ + "_" + cam_channel_ );
        std::cout << envoy_opts << " | " << RedisHelper::get_redis_uri() <<  std::endl;
        envoy_ = std::make_shared<RedisEnvoy>(envoy_opts);
        ROS_WARN("echo: %s", envoy_->echo("Redis connected").c_str());

        // Phase one Initialization
        P1::ImageSdk::Initialize();
        P1::ImageSdk::SetSensorProfilesLocation(
        "/root/kamera_ws/build/phase_one/ImageSDK/SensorProfiles");
        ROS_INFO_STREAM("Setting number of processing threads to " << num_threads_ << ".");
        P1::ImageSdk::SetThreadPoolThreadCount(num_threads_);
        connectToIPCamera(ip_address_);

        auto list = camera.AllPropertyIds();
        for (auto propertyId : list) {
            P1::CameraSdk::PropertySpecification property =
                            camera.PropertySpec(propertyId);
            std::string name = property.mName;
            auto val = camera.Property(propertyId);
            std::cout << name << " | " << val.ToString() << "\n";
        }
        getPropertyMaps(camera, property_to_id_, property_to_type_);

        static double min_image_delay = 0.95; // should depend on exposure time
        event_cache.set_delay(min_image_delay);
        event_cache.set_tolerance(ros::Duration(0.49));
        std::string to_process_filename = "/mnt/data/to_process.txt";
        std::string processed_filename = "/mnt/data/processed.txt";
        std::vector<std::string> to_process_images = loadFile(to_process_filename);
        std::vector<std::string> processed_images = loadFile(processed_filename);
        // write out any files that are left, as well as add to queue
        std::ofstream write_intersection;
        write_intersection.open(to_process_filename);
        for (auto& toProcess : to_process_images) {
            if (std::find(processed_images.begin(), processed_images.end(),
                    toProcess) != processed_images.end()) {
            } else {
                write_intersection << toProcess << std::endl;
                filename_q_.push(toProcess);
            }
        }
        write_intersection.close();
        // Just delete processed image file
        remove(processed_filename.c_str());
        // Append new entries to cache
        to_process_out.open(to_process_filename, std::ios_base::app);
        processed_out.open(processed_filename, std::ios_base::app);
            
        // Throw results into redis for access
        std::string base = "/sys/" + hostname + "/p1debayerq/";
        std::string total = base + "total";
        std::string num_processed = base + "processed";
        std::string processed_str = std::to_string(processed_counter);
        std::string total_str = std::to_string(total_counter);
        envoy_->put(total, total_str);
        envoy_->put(num_processed, processed_str);

        running_ = true;
        capture_thread_ = std::thread(&PhaseOne::capture, this);
        demosaic_thread_ = std::thread(&PhaseOne::demosaic, this);

        get_param_service_ = pnh.advertiseService("get_phaseone_parameter",
                &PhaseOne::getPhaseOneParameter, this);
        set_param_service_ = pnh.advertiseService("set_phaseone_parameter",
                &PhaseOne::setPhaseOneParameter, this);
        image_view_service_ = pnh.advertiseService("get_image_view",
                &PhaseOne::getImageView, this);
        compressed_image_view_service_ = pnh.advertiseService("get_compressed_image_view",
                &PhaseOne::getCompressedImageView, this);

        // Subscribers
        event_sub_ = pnh.subscribe("/event", 1, &PhaseOne::eventCallback, this);
    };

    int PhaseOne::connectToIPCamera(std::string ip) {
        ROS_INFO_STREAM("Attempting to connect to IP connected camera at "
                        << ip << "...");
        try {
            this->camera = P1::CameraSdk::Camera::OpenIpCamera(ip);
        }
        catch (P1::ImageSdk::SdkException exception)
        {
            // Exception from ImageSDK
            std::cout << "ImageSDK Exception: " << exception.what() <<
                            " Code:" << exception.mCode << std::endl;
            if (running_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                return connectToIPCamera(ip);
            } else
                return 1;
        }
        catch (P1::CameraSdk::SdkException exception)
        {
            // Exception from CameraSDK
            std::cout << "CameraSDK Exception: " << exception.what() << " Code:"
                        << exception.mErrorCode << std::endl;
            if (running_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                return connectToIPCamera(ip);
            } else
                return 1;
        }
        catch (...)
        {
            // Any other exception - just in case
            std::cout << "Argh - we got an exception" << std::endl;
            if (running_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                return connectToIPCamera(ip);
            } else
                return 1;
        }

        ROS_INFO("IP Camera connected!");
        return 0;
    };

    int PhaseOne::capture() {
        ROS_INFO("|CAPTURE| thread started");
        P1::CameraSdk::Listener imgListener;
        imgListener.EnableNotification(camera,
                P1::CameraSdk::EventType::CameraImageReady);
        camera.Subscriptions()->FullImages()->Subscribe();
        // Necessary to sleep, otherwise TriggerCapture
        // will start before EnableImageReceiving is done
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        bool timed_out = false;
        try {
            while (running_) {
                auto tic1 = std::chrono::high_resolution_clock::now();
                if (trigger_mode_ == "auto") {
                    ROS_INFO("|CAPTURE| Triggering camera capture");
                    camera.TriggerCapture();
                } // else wait for image (hardware signal)

                ROS_INFO("|CAPTURE| Waiting for image from camera...");
                // Create stat msg
                auto nodeName = "/" + hostname + "/rgb/rgb_driver";
                custom_msgs::Stat stat_msg;
                std::stringstream link;
                stat_msg.header.stamp = ros::Time::now();
                stat_msg.trace_topic = nodeName + "/publishImage";
                stat_msg.node = nodeName;
                link << nodeName << "/event/" << event_.header.seq; // link this trace to the event trace
                stat_msg.link = link.str();

                // Wait for image
                P1::CameraSdk::NotificationEventPtr event =
                                imgListener.WaitForNotification(15000);
                std::shared_ptr<const P1::CameraSdk::IFullImage> iiqImage;
                P1::ImageSdk::RawImage image;
                ros::Time frame_recv_time_;
                if ( event && event->Type().id == P1::CameraSdk::EventType::CameraImageReady.id ) {
                    frame_recv_time_ = ros::Time::now();
                    ROS_INFO("|CAPTURE| image received!");
                    iiqImage = event->FullImage();
                    P1::ImageSdk::RawImage image(iiqImage->Data(),
                            iiqImage->DataSizeBytes());
                    std::unique_lock<std::mutex> lock(mtx);
                    ROS_INFO("|CAPTURE| Entered lock!");
                    if ( !image_q_.empty() ) {
                        image_q_.pop();
                        image_q_.push(image);
                    } else {
                        image_q_.push(image);
                    }
                    new_image = true;
                    ROS_INFO_STREAM("|CAPTURE| image queue size: " << image_q_.size());
                    // mutex locks goes out of scope, should release
                } else {
                    ROS_WARN("|CAPTURE| No image received within timeout!");
                    //running_ = false;
                    timed_out = true;
                    break;
                    //continue;
                }
                auto tic3 = std::chrono::high_resolution_clock::now();
                    
                // Sync event cache
                std_msgs::Header gps_header;
                ROS_INFO("|CAPTURE| Searching event cache.");
                bool success = event_cache.search(frame_recv_time_, gps_header);
                if (success != true) {
                    ROS_WARN("|CAPTURE| Could not find event message to associate image to! Skipping.");
                    // TODO: sub in current time? seems dangerous
                    //gps_header.stamp = ros::Time::now();
                    continue;
                } else {
                    ROS_WARN("|CAPTURE| Found matching event!");
                    stat_msg.note = "success";
                }   
                ROS_INFO("|CAPTURE| Checking if archiving.");
                int is_archiving = ArchiverHelper::get_is_archiving(envoy_, "/sys/arch/is_archiving");
                if (is_archiving) {
                    // Write raw iiq to file
                    long int sec  = gps_header.stamp.sec;
                    long int nsec = gps_header.stamp.nsec;
                    std::string fname = ArchiverHelper::generateFilename(envoy_, arch_opts_, sec, nsec);
                    std::string effort = envoy_->get("/sys/arch/project");
                    // Adds an additional dir layer for raw images
                    fname.insert(fname.find(effort), "iiq_buffer/");
                    //std::string fname = "/mnt/data/test/" + iiqImage->FileName();
                    fname.replace(fname.find("jpg"), 3, "IIQ"); // Replaces "jpg" with "IIQ"
                    ROS_INFO_STREAM("|CAPTURE| Write file: " << fname);
                    boost::filesystem::path path_fname{fname};
                    try {
                        boost::filesystem::create_directories(path_fname.parent_path());
                        std::fstream iiqFile(fname, std::ios::binary | std::ios::trunc | std::ios::out);
                        iiqFile.write((char*)iiqImage->Data().get(), iiqImage->DataSizeBytes());
                        iiqFile.close();
                        to_process_out << fname << std::endl;
                        std::unique_lock<std::mutex> lock(mtx);
                        filename_q_.push(fname);
                        total_counter++;
                    } catch (boost::filesystem::filesystem_error &e) {
                        ROS_ERROR("|CAPTURE| Archive Failed [%d]: %s", e.code().value(), e.what());
                    }
                }

                //P1::ImageSdk::TagId ids;
                //P1::ImageSdk::ImageTag tag = image.GetTag(ids.DateTime);
                //ROS_INFO_STREAM("Tag: " << tag.ToString());
                // seems to want to take from queue, rather than straight `image`
                P1::ImageSdk::RawImage raw_img;
                raw_img = image_q_.front();
                P1::ImageSdk::BitmapImage preview = raw_img.GetPreview();
                cv::Mat cvImage_raw = cv::Mat(cv::Size(preview.Width(), preview.Height()), CV_8UC3);
                cvImage_raw.data = preview.Data().get();
                sensor_msgs::Image output_msg;
                try {
                    std_msgs::Header header;
                    //P1::ImageSdk::TagId ids;
                    //P1::ImageSdk::ImageTag tag = raw_img.GetTag(ids.CaptureNumber);
                    //header.seq = tag.Value();
                    header.seq = gps_header.seq;
                    header.stamp = gps_header.stamp;
                    img_bridge = cv_bridge::CvImage(header,
                            sensor_msgs::image_encodings::RGB8, cvImage_raw);
                    img_bridge.toImageMsg(output_msg);
                } catch (...) {
                    ROS_ERROR("|CAPTURE| Failed to convert preview.");
                };
                image_pub.publish(output_msg);
                stat_pub_.publish(stat_msg);
                //long int sec  = output_msg.header.stamp.sec;
                //long int nsec = output_msg.header.stamp.nsec;
                //std::string fname = ArchiverHelper::generateFilename(envoy_, arch_opts_, sec, nsec);

                auto toc = std::chrono::high_resolution_clock::now();
                auto dt = toc - tic1;
                ROS_INFO_STREAM("|CAPTURE| total time: " << dt.count() / 1e9 << "s");
                if (trigger_mode_ == "auto") {
                    int trigger_sleep = (int) 1000 * (1 / auto_trigger_rate_);
                    std::this_thread::sleep_for(std::chrono::milliseconds(trigger_sleep));
                }
            } // capture loop
            if (timed_out) {
                // failed to get an image, something might be up, or
                // trigger might not be running. Close connection and try again.
                // This helps keep alive the debayer thread, so even if no images are coming
                // in it will keep processing until killed
                ROS_WARN("|CAPTURE| Timed out: closing camera connection and retrying.");
                if ( !image_q_.empty() )
                    image_q_.pop();
                // Empty queue before closing camera
                camera.Close();
                int ret = connectToIPCamera(ip_address_);
                ret = capture();
            }
        }// catch-all for exceptions during capture
        catch (P1::ImageSdk::SdkException exception)
        {
            ROS_ERROR_STREAM("|CAPTURE| ImageSDK Exception: " << exception.what() << " Code:"
                                << exception.mCode << std::endl);
            running_ = false;
            return 1;
        }
        catch (P1::CameraSdk::SdkException exception)
        {
            ROS_ERROR_STREAM("|CAPTURE| CameraSDK Exception: " << exception.what() << " Code:"
                                << exception.mErrorCode << std::endl);
            running_ = false;
            return 1;
        }
        catch (const std::exception& e) {
            ROS_ERROR_STREAM("|CAPTURE| Caught generic exception: " << e.what());
            return 1;
        }
        catch (...)
        {
            std::cout << "|CAPTURE| Argh - we got an exception" << std::endl;
            running_ = false;
            return 1;
        }
        return 0;
    };

    int PhaseOne::demosaic() {
        ROS_INFO("|DEMOSAIC| thread started");
        while (running_) {
            auto tic1 = std::chrono::high_resolution_clock::now();
            std::string filename;
            std::unique_lock<std::mutex> lock(mtx);
            if ( !filename_q_.empty() ) {
                // grab most-recent filename (could make a LIFO queue?)
                filename = filename_q_.front();
                filename_q_.pop();
            } else {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }
            lock.unlock();
            ROS_INFO_STREAM("|DEMOSAIC| read file: " << filename);
            auto tic2 = std::chrono::high_resolution_clock::now();
            P1::ImageSdk::RawImage imageFile(filename);
            P1::ImageSdk::BitmapImage bitmap;
            try {
                // demosaic image into RGB format from IIQ
                std::lock_guard<std::mutex> lock(debayer_mtx);
                bitmap = convertConfig.ApplyTo(imageFile);
            }
            catch (P1::ImageSdk::SdkException exception)
            {
                // Exception from ImageSDK
                ROS_ERROR_STREAM("|DEMOSAIC| ImageSDK Exception: " << exception.what() << " Code:"
                                << exception.mCode << std::endl);
                continue;
            }
            catch (P1::CameraSdk::SdkException exception)
            {
                // Exception from CameraSDK
                ROS_ERROR_STREAM("|DEMOSAIC| CameraSDK Exception: " << exception.what() << " Code:"
                                << exception.mErrorCode << std::endl);
                continue;
            }
            catch (...)
            {
                // Any other exception - just in case
                ROS_WARN("|DEMOSAIC| Argh - we got an exception in debayering.");
                continue;
            }
            auto toc = std::chrono::high_resolution_clock::now();
            auto dt = toc - tic2;
            ROS_INFO_STREAM("|DEMOSAIC| load and demosaic time: " << dt.count() / 1e9 << "s");
            std::string filename_iiq(filename);
            filename.replace(filename.find("IIQ"), 3, "jpg"); // Replaces "IIQ" with "jpg"
            filename.erase(filename.find("iiq_buffer/"), 11); // Removes buffer dir from path
            ROS_INFO_STREAM("|DEMOSAIC| write file: " << filename);
            bool success = dumpImage(imageFile, bitmap, filename, "jpg");
            if ( success ) {
                // delete IIQ file that's been processed
                remove(filename_iiq.c_str());
                processed_out << filename_iiq << std::endl;
                processed_counter++;
            } else {
                ROS_ERROR_STREAM("IIQ File: " << filename_iiq << " failed to save to disk as jpg.");
            }
            // Throw results into redis for access
            std::string base = "/sys/" + hostname + "/p1debayerq/";
            std::string total = base + "total";
            std::string num_processed = base + "processed";
            std::string processed_str = std::to_string(processed_counter);
            std::string total_str = std::to_string(total_counter);
            envoy_->put(total, total_str);
            envoy_->put(num_processed, processed_str);
            toc = std::chrono::high_resolution_clock::now();
            dt = toc - tic1;
            ROS_INFO_STREAM("|DEMOSAIC| total time: " << dt.count() / 1e9 << "s");
        }
        return 0;
    };

    bool PhaseOne::dumpImage(P1::ImageSdk::RawImage rawImage,
                             P1::ImageSdk::BitmapImage bitmap,
                             const std::string &filename,
                             std::string format)
    {
        auto start_db  = ros::Time::now();
        try {
            boost::filesystem::path path_filename{filename};
            boost::filesystem::create_directories(path_filename.parent_path());
            if (format == "jpg") {
                int quality = std::stoi(envoy_->get("/sys/arch/jpg/quality"));
                jpegConfig.quality = quality;
                P1::ImageSdk::JpegWriter(filename, bitmap, rawImage, jpegConfig);
            } else if (format == "tiff") {
                P1::ImageSdk::TiffConfig tiffConfig;
                P1::ImageSdk::TiffWriter(filename, bitmap, rawImage, tiffConfig);
            }
            return true;
        } catch (P1::ImageSdk::SdkException exception) {
            ROS_ERROR_STREAM("|DUMP| ImageSDK Exception: " << exception.what() << " Code:"
                            << exception.mCode << std::endl);
            return false;
        }
        catch (const std::exception& e) {
            ROS_ERROR_STREAM("|DUMP| Caught generic exception: " << e.what());
            return false;
        }
        catch (...) {
            std::cout << "|DUMP| Argh - we got an exception" << std::endl;
            return false;
        }
    }

    void PhaseOne::getPropertyMaps(const P1::CameraSdk::Camera& camera,
                                   std::map<std::string, int>& property_to_id_,
                                   std::map<std::string, P1::CameraSdk::PropertyValue>& property_to_type) {
        ROS_INFO("Grabbing properties from camera.");
        std::vector<uint32_t> propertyIdList = camera.AllPropertyIds();
        for (int propertyId : propertyIdList)
        {
            // Get the property specification for the current propertyId
            P1::CameraSdk::PropertySpecification property = camera.PropertySpec(propertyId);
            std::string name = property.mName;
            property_to_id_[name] = propertyId;
            property_to_type_[name] = property.mValue;
        }
    };

    bool PhaseOne::getPhaseOneParameter(phase_one::GetPhaseOneParameter::Request& req,
                                        phase_one::GetPhaseOneParameter::Response& resp)
    {
        if (property_to_id_.size() > 0)
        {
            try
            {
                std::string name = req.name.c_str();
                int id = property_to_id_[ name ];
                P1::CameraSdk::PropertyValue pv = camera.Property(id);
                resp.value = pv.ToString();
                resp.message = "ok";
                ROS_INFO_STREAM("|GET| Parameter " << name << " returning value " << resp.value);
            }
            catch (const std::exception& ex)
            {
                ROS_ERROR_STREAM("Cannot get parameter: " << ex.what());
                resp.message = ex.what();
            }
        }

        return true;

    };

    bool PhaseOne::setPhaseOneParameter(phase_one::SetPhaseOneParameter::Request& req,
                                        phase_one::SetPhaseOneParameter::Response& resp)
    {
        if (property_to_id_.size() > 0)
        {
            try
            {
                std::string parameters = req.parameters.c_str();
                std::map<std::string, std::string> name_to_value =
                                                    parseParams(parameters);

                for (auto const& it: name_to_value) {
                    std::string name = it.first;
                    std::string val = it.second;
                    int id = property_to_id_[ name ];
                    auto cam_val = camera.Property(id);
                    ROS_INFO_STREAM("|SET| Setting '" << name << "' from '"
                                    << cam_val.ToString() << "' to '" << val << "'");
                    std::string s1 = property_to_type_[ name ].ToString();
                    if (s1.find("Enum") != std::string::npos ||
                            s1.find("Bool") != std::string::npos ||
                            s1.find("Int") != std::string::npos) {
                        P1::CameraSdk::PropertyValue pv;
                        pv.mType = property_to_type_[name].mType;
                        pv.mInt = std::stoi(val);
                        camera.SetProperty(id, pv);
                    } else if (s1.find("Float") != std::string::npos) {
                        P1::CameraSdk::PropertyValue pv;
                        pv.mType = property_to_type_[name].mType;
                        pv.mDouble = std::stod(val);
                        camera.SetProperty(id, pv);
                    } else {
                        P1::CameraSdk::PropertyValue pv;
                        pv.mType = property_to_type_[name].mType;
                        pv.mString = val;
                        camera.SetProperty(id, pv);
                    }
                }
            }
            catch (const std::exception& ex)
            {
                ROS_ERROR_STREAM("Cannot set parameter: " << ex.what());
                resp.message = ex.what();
                return false;
            }
            resp.message = "ok";
        }
        return true;
    };


    bool PhaseOne::getCompressedImageView(phase_one::GetCompressedImageView::Request& req,
                                            phase_one::GetCompressedImageView::Response& resp) {
        auto tic = std::chrono::high_resolution_clock::now();
        ROS_INFO("Received Request for Compresssed Image View.");
        P1::ImageSdk::RawImage raw_img;
        std::unique_lock<std::mutex> lock(mtx);
        if ( !image_q_.empty() ) {
            raw_img = image_q_.front();
        } else {
            resp.success = false;
            return false;
        }
        // manually release lock early
        lock.unlock();
        // Extract destination points from source image given a generic homography
        std::vector<double> H = req.homography;
        // Construct proper dimensional homography
        cv::Mat warp_matrix;
        warp_matrix = cv::Mat::eye(3, 3, CV_32F);
        int row = 0;
        int col = 0;
        for (auto &hom : H)
        {   // stuff the values into the matrix
            warp_matrix.at<float>(row, col) = hom;
            col++;
            if (col > 2)
            {
                col = 0;
                row++;
        } }
        float raw_width = raw_img.Width();
        float raw_height = raw_img.Height();
        std::vector<cv::Point2f> srcPoints = {{0, 0}, {raw_width, 0}, {raw_width, raw_height}, {0, raw_height}};
        std::vector<cv::Point2f> dstPoints;
        try {
            cv::perspectiveTransform(srcPoints, dstPoints, warp_matrix);
        } catch(...) {
            ROS_ERROR("CV Warp exception in translating homography to points.");
            resp.success = false;
            return false;
        }
        int x = dstPoints[0].x;
        int y = dstPoints[0].y;
        int h = req.output_height;
        int w = req.output_width;

        P1::ImageSdk::ConvertConfig config;
        config.SetCrop(x, y, w, h);
        auto tic1 = std::chrono::high_resolution_clock::now();
        P1::ImageSdk::BitmapImage bitmap;
        try {
            // demosaic image into RGB format from IIQ
            std::lock_guard<std::mutex> lock(debayer_mtx);
            bitmap = convertConfig.ApplyTo(raw_img);
        }
        catch (P1::ImageSdk::SdkException exception)
        {
            // Exception from ImageSDK
            ROS_ERROR_STREAM("|DEMOSAIC| ImageSDK Exception: " << exception.what() << " Code:"
                            << exception.mCode << std::endl);
            return false;
        }
        catch (P1::CameraSdk::SdkException exception)
        {
            // Exception from CameraSDK
            ROS_ERROR_STREAM("|DEMOSAIC| CameraSDK Exception: " << exception.what() << " Code:"
                            << exception.mErrorCode << std::endl);
            return false;
        }
        catch (...)
        {
            // Any other exception - just in case
            ROS_WARN("|DEMOSAIC| Argh - we got an exception in debayering.");
            return false;
        }
        ROS_INFO_STREAM(bitmap.Height() << " bitmap " << bitmap.Width() << "\n");
        auto toc1 = std::chrono::high_resolution_clock::now();
        auto dt1 = toc1 - tic1;
        ROS_INFO_STREAM("View Server: Time to convert image is: " << dt1.count() / 1e9 << "s\n");
        cv::Mat cvImage_raw = cv::Mat(cv::Size(bitmap.Width(), bitmap.Height()), CV_8UC3);
        cvImage_raw.data = bitmap.Data().get();

        try {
            std_msgs::Header header;
            header.seq = 0;
            header.stamp = ros::Time::now();
            img_bridge = cv_bridge::CvImage(header,
                    sensor_msgs::image_encodings::RGB8, cvImage_raw);
            sensor_msgs::CompressedImage output_msg;
            output_msg.format = "jpg";
            output_msg.header = header;
            int quality = std::stoi(envoy_->get("/sys/arch/jpg/quality"));
            std::vector<int> compression_params;
            compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
            compression_params.push_back(quality);
            std::vector<uchar> buffer;
            int MB = 1024 * 1024;
            buffer.resize(100 * MB);
            ROS_INFO("Calling imencode");
            cv::imencode(".jpg", cvImage_raw, buffer, compression_params);
            ROS_INFO("Calling tocompressimage");
            output_msg.data = buffer;
            img_bridge.toCompressedImageMsg(output_msg);
            resp.success = true;
            resp.image = output_msg;
        } catch (...) {
            ROS_ERROR("CV Warp exception.");
            resp.success = false;
            return false;
        };
        return true;
        auto toc = std::chrono::high_resolution_clock::now();
        auto dt = toc - tic;
        ROS_INFO_STREAM("View Server: Time to process image request was: " << dt.count() / 1e9 << "s\n");
        return true;
    };

    bool PhaseOne::getImageView(custom_msgs::RequestImageView::Request& req,
                                custom_msgs::RequestImageView::Response& resp) {
        auto tic = std::chrono::high_resolution_clock::now();
        ROS_INFO("|VIEW SERVER| : Received Request for image view.");
        P1::ImageSdk::RawImage raw_img;
        std::unique_lock<std::mutex> lock(mtx);
        bool staleH = req.homography == lastH; 
        bool staleSat = req.show_saturated_pixels == last_show_sat; 
        lastH = req.homography;
        last_show_sat = req.show_saturated_pixels;
        if ( staleH && staleSat && !new_image) {
            resp.success = true;
            resp.image = sensor_msgs::Image();
            return true;
        }
        if ( !image_q_.empty() ) {
            raw_img = image_q_.front();
        } else {
            resp.success = false;
            resp.image = sensor_msgs::Image();
            return false;
        }
        // Reset global tracking of a "new image" received
        new_image = false;
        // manually release lock early
        lock.unlock();
        // Extract destination points from source image given a generic homography
        std::vector<double> H = req.homography;
        // Construct proper dimensional homography
        cv::Mat warp_matrix;
        warp_matrix = cv::Mat::eye(3, 3, CV_32F);
        int row = 0;
        int col = 0;
        for (double &hom : H)
        {   // stuff the values into the matrix
            warp_matrix.at<float>(row, col) = hom;
            col++;
            if (col > 2)
            {
                col = 0;
                row++;
        } }
        float raw_width = raw_img.Width();
        float raw_height = raw_img.Height();
        std::vector<cv::Point2f> srcPoints = {{0, 0}, {raw_width, 0}, {raw_width, raw_height}, {0, raw_height}};
        std::vector<cv::Point2f> dstPoints;
        try {
            cv::perspectiveTransform(srcPoints, dstPoints, warp_matrix);
        } catch(...) {
            ROS_ERROR("|VIEW SERVER| : CV Warp exception in translating homography to points.");
            resp.success = false;
            return false;
        }
        int x = dstPoints[0].x;
        int y = dstPoints[0].y;
        int out_h = req.output_height;
        int out_w = req.output_width;
        double scale = H[0];
        double float_h = scale * out_h;
        double float_w = scale * out_w;
        // Cast afterwards so early rounding doesn't occur
        int h = (int) float_h;
        int w = (int) float_w;
        // Bounds checking
        if (x < 0) x = 0; if (x > raw_img.Width()) x = raw_img.Width();
        if (y < 0) y = 0; if (y > raw_img.Height()) y = raw_img.Height();
        if (h < 0) h = 0; if (h > raw_img.Height() - y) h = raw_img.Height() - y;
        if (w < 0) w = 0; if (w > raw_img.Width() - x) w = raw_img.Width() - x;
        // All this gets a nice crop out of the image quite quickly (0.2s at worst)
        P1::ImageSdk::ConvertConfig config;
        config.SetCrop(x, y, w, h);
        config.SetOutputHeight(out_h);
        config.SetOutputWidth(out_w);
        config.SetOutputFormat(P1::ImageSdk::BitmapFormat::Rgb24);
        auto tic1 = std::chrono::high_resolution_clock::now();
        P1::ImageSdk::BitmapImage bitmap;
        try {
            // demosaic image into RGB format from IIQ
            std::lock_guard<std::mutex> lock(debayer_mtx);
            bitmap = config.ApplyTo(raw_img);
        }
        catch (P1::ImageSdk::SdkException exception)
        {
            // Exception from ImageSDK
            ROS_ERROR_STREAM("|VIEW SERVER| ImageSDK Exception: " << exception.what() << " Code:"
                            << exception.mCode << std::endl);
            resp.success = false;
            return false;
        }
        catch (P1::CameraSdk::SdkException exception)
        {
            // Exception from CameraSDK
            ROS_ERROR_STREAM("|VIEW SERVER| CameraSDK Exception: " << exception.what() << " Code:"
                            << exception.mErrorCode << std::endl);
            resp.success = false;
            return false;
        }
        catch (...)
        {
            // Any other exception - just in case
            ROS_WARN("|VIEW SERVER| Argh - we got an exception in debayering.");
            resp.success = false;
            return false;
        }
        auto toc1 = std::chrono::high_resolution_clock::now();
        auto dt1 = toc1 - tic1;
        ROS_INFO_STREAM("|VIEW SERVER| : Time to convert image is: " << dt1.count() / 1e9 << "s\n");
        cv::Mat cvImage_raw = cv::Mat(cv::Size(bitmap.Width(), bitmap.Height()), CV_8UC3);
        cvImage_raw.data = bitmap.Data().get();
        try {
            if (req.show_saturated_pixels) {
                int maxval = 255;
                cv::Scalar sat_pix = cv::Scalar(maxval, maxval, maxval);
                cv::Mat mask;
                cv::inRange(cvImage_raw, sat_pix, sat_pix, mask);
                // Set white pixels to red
                cvImage_raw.setTo(cv::Scalar(255, 0, 0), mask);
            }
        } catch (const std::exception& e) {
            ROS_ERROR_STREAM("|VIEW SERVER| Caught generic exception in sat pixels: " << e.what());
            return false;
        }
        try {
            std_msgs::Header header;
            header.seq = 0;
            header.stamp = ros::Time::now();
            img_bridge = cv_bridge::CvImage(header,
                    sensor_msgs::image_encodings::RGB8, cvImage_raw);
            sensor_msgs::Image output_msg;
            img_bridge.toImageMsg(output_msg);
            resp.success = true;
            resp.image = output_msg;
        } catch (...) {
            ROS_ERROR("|VIEW SERVER| : CV Warp exception.");
            resp.success = false;
            return false;
        };
        auto toc = std::chrono::high_resolution_clock::now();
        auto dt = toc - tic;
        ROS_INFO_STREAM("|VIEW SERVER| : Time to process image request was: " << dt.count() / 1e9 << "s\n");
        return true;
    };

    void PhaseOne::eventCallback (const boost::shared_ptr<custom_msgs::GSOF_EVT const>& msg) {
        ROS_INFO("[%d]<1> eventCallback <>         %2.2f", msg->header.seq, msg->header.stamp.toSec());
        event_ = *msg;
        event_cache.push_back(msg->sys_time, msg);
        // TODO: hardcoded, but is OK for now
        auto nodeName = "/" + hostname + "/rgb/rgb_driver";
        custom_msgs::Stat stat_msg;
        std::stringstream link;
        stat_msg.header.stamp = ros::Time::now();
        stat_msg.trace_header = (*msg).header;
        stat_msg.trace_topic = nodeName + "/eventCallback";
        stat_msg.node = nodeName;
        link << nodeName << "/event/" << event_.header.seq; // link this trace to the event trace
        stat_msg.link = link.str();
        stat_pub_.publish(stat_msg);
        event_cache.purge();
        //watchdog.check();
        //event_cache.show();
    };
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "phase_one_standalone");
    phase_one::PhaseOne cls;
    cls.init();
    ros::Rate r(10);
    while (cls.running_) {
        ros::spinOnce();
        r.sleep();
    }
    return 0;
}
