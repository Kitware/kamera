#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <thread>
#include <boost/filesystem.hpp>

#include <P1Camera.hpp>
#include <P1Image.hpp>
#include <P1ImageJpegWriter.hpp>
#include <P1ImageTiffWriter.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Header.h>

#include <phase_one/GetPhaseOneParameter.h>
#include <phase_one/SetPhaseOneParameter.h>

using namespace P1::CameraSdk;
using namespace P1::ImageSdk;

using std::cout;
using std::endl;

namespace phase_one
{
    class PhaseOneNodelet : public nodelet::Nodelet
    {
        public:
            PhaseOneNodelet()
            {
            }

        private:
            bool running_ = false;
            int counter = 0;
            std::thread capture_thread_;
            image_transport::Publisher image_pub;
            cv_bridge::CvImage img_bridge;
            P1::CameraSdk::Camera camera;
            P1::ImageSdk::DecodeConfig decodeConfig;
            P1::ImageSdk::ConvertConfig convertConfig;
            P1::ImageSdk::JpegConfig jpegConfig;
            P1::CameraSdk::Listener listener;
            ros::ServiceServer get_param_service_;
            ros::ServiceServer set_param_service_;
            std::map<std::string, int> property_to_id_;
            std::map<std::string, P1::CameraSdk::PropertyValue> property_to_type_;


        ~PhaseOneNodelet() {
          ROS_INFO("Phase One Camera: Shutting down");
          // signal running_ threads and wait until they finish
          running_ = false;
          if (capture_thread_.joinable())
          {
            capture_thread_.join();
          }
          // Disable image receiving
          camera.Close();
        };

        virtual void onInit() {
            ROS_INFO("PhaseOne Driver: Initialization");
            // ROS initialization
            ros::NodeHandle& nh = getNodeHandle();
            ros::NodeHandle& pnh = getPrivateNodeHandle();
            image_transport::ImageTransport it(nh);
            image_pub = it.advertise("image_raw", 1);

            // Phase one Initialization
            P1::ImageSdk::Initialize();
            P1::ImageSdk::SetSensorProfilesLocation(
                    "/home/user/phaseOne/phase_one_ws/build/phase_one/ImageSDK/SensorProfiles");
            get_param_service_ = pnh.advertiseService("get_phaseone_parameter",
                    &PhaseOneNodelet::getPhaseOneParameter, this);
            set_param_service_ = pnh.advertiseService("set_phaseone_parameter",
                    &PhaseOneNodelet::setPhaseOneParameter, this);
            connectToIPCamera("192.168.1.6");
            auto list = camera.AllPropertyIds();
            for (auto propertyId : list) {
            	P1::CameraSdk::PropertySpecification property = camera.PropertySpec(propertyId);
                std::string name = property.mName;
                auto val = camera.Property(propertyId);
                std::cout << name << " | " << val.ToString() << "\n";
            }
            //connectToUSBCamera();
            getPropertyMaps();

            running_ = true;
	    //config.SetCrop(5000, 5000, 1920, 1080);
	    //config.SetOutputHeight(0.1)
            capture_thread_ = std::thread(&PhaseOneNodelet::capture, this);
        };

        int connectToUSBCamera() {
            cout << "Probing for available USB cameras..." << endl;
            std::vector<CameraListElement> list;
            try {
                list = Camera::AvailableCameras();
                for(auto cam : list)
                {
                    ROS_INFO_STREAM(" * " << cam.mName << " ("
                                    << cam.mSerialNum << ")");
                }
            }
            catch (P1::ImageSdk::SdkException exception)
	        {
	        	// Exception from ImageSDK
	        	ROS_ERROR_STREAM("ImageSDK Exception: " << exception.what()
                                 << " Code:" << exception.mCode);
	        	return -1;
	        }
	        catch (P1::CameraSdk::SdkException exception)
	        {
	        	// Exception from CameraSDK
	        	std::cout << "CameraSDK Exception: " << exception.what() << " Code:"
	        		<< exception.mErrorCode <<
	        		std::endl;
	        	return -1;
	        }
	        catch (...)
	        {
	        	// Any other exception - just in case
	        	std::cout << "Argh - we got an exception" << std::endl;
	        	return -1;
	        }

            if (list.size() <= 0)
            {
                cout << "Could not find any USB cameras!" << endl;
                return -1;
            } else {
                ROS_INFO_STREAM("Found " << list.size() << " camera"
                                << (list.size() == 1 ? "" : "s"));
                this->camera = P1::CameraSdk::Camera::OpenUsbCamera();
                return 0;
            }
        };

        int connectToIPCamera(std::string ip) {
            ROS_INFO_STREAM("Attempting to connect to IP connected camera at " << ip << "...");
            try {
                this->camera = P1::CameraSdk::Camera::OpenIpCamera(ip);
            }
            catch (P1::ImageSdk::SdkException exception)
	        {
	        	// Exception from ImageSDK
	        	std::cout << "ImageSDK Exception: " << exception.what() << " Code:" << exception.mCode <<
	        		std::endl;
	        	return -1;
	        }
	        catch (P1::CameraSdk::SdkException exception)
	        {
	        	// Exception from CameraSDK
	        	std::cout << "CameraSDK Exception: " << exception.what() << " Code:"
	        		<< exception.mErrorCode <<
	        		std::endl;
	        	return -1;
	        }
	        catch (...)
	        {
	        	// Any other exception - just in case
	        	std::cout << "Argh - we got an exception" << std::endl;
	        	return -1;
	        }

            ROS_INFO("IP Camera connected!");
            return 0;
        };

        int capture() {
            ROS_INFO("Capture thread started");
            P1::CameraSdk::Listener imgListener;
            imgListener.EnableNotification(camera,
                    P1::CameraSdk::EventType::CameraImageReady);
            camera.Subscriptions()->FullImages()->Subscribe();
            // Necessary to sleep, otherwise TriggerCapture will start before EnableImageReceiving is done
		std::this_thread::sleep_for(std::chrono::milliseconds(200));
            try {
                while (running_) {
                    auto tic1 = std::chrono::high_resolution_clock::now();
                    ROS_INFO("Triggering camera capture");
                    camera.TriggerCapture();

                    ROS_INFO("Waiting for image from camera...");
                    // Wait for image
                    P1::CameraSdk::NotificationEventPtr event =
                                   imgListener.WaitForNotification(10000);
                    ROS_INFO("Image received!");
                    auto iiqImage = event->FullImage();

                    // Write RGB bitmap to file
		            //std::fstream iiqFile("bits.iiq", std::ios::binary | std::ios::trunc |
		            //	std::ios::out);
		            //iiqFile.write((char*)imageFile.Data.get(), imageFile.size);
		            //iiqFile.close();
                    auto toc = std::chrono::high_resolution_clock::now();
                    auto dt = toc - tic1;
                    ROS_INFO_STREAM("Time to fetch image off camera was: " << dt.count() / 1e9 << "s");


                    auto tic2 = std::chrono::high_resolution_clock::now();
                    //P1::ImageSdk::RawImage new_image("bits.iiq");
                    P1::ImageSdk::RawImage image(iiqImage->Data(),
                            iiqImage->DataSizeBytes());
                    //P1::ImageSdk::SensorBayerOutput decodedImage =
                    //              image.Decode(decodeConfig);
                    //P1::ImageSdk::SensorBayerOutput decodedImage =
                    //             image.Decode(decodeConfig);
                    //std::cout << "Decoded image size: " << decodedImage.ByteSize()
                    //    << " Height=" << decodedImage.FullHeight()
                    //    << " Width=" << decodedImage.FullWidth() << std::endl;
                    // convertConfig.WhiteBalanceGainRange
                    P1::ImageSdk::BitmapImage bitmap = convertConfig.ApplyTo(image);
                    toc = std::chrono::high_resolution_clock::now();
                    dt = toc - tic2;
                    ROS_INFO_STREAM("Time to demosaic iiq to raw was: "
                                    << dt.count() / 1e9 << "s");

                    ROS_INFO_STREAM("Received Image Height=" << bitmap.Height() << " Width= "
		            	<< bitmap.Width() << "Size= " << bitmap.ByteSize() << std::endl);

		            // Toss raw image in cv mat
                    auto tic3 = std::chrono::high_resolution_clock::now();
                    //cv::Mat cvImage_raw = cv::Mat(cv::Size(
                    //                 bitmap.Width(), bitmap.Height()), CV_16UC3);
                    //cout << "Dumping data into mat.\n";
                    //cvImage_raw.data = bitmap.Data().get();
                    //cv::Mat cvImage = cv::Mat(cv::Size(
                    //                bitmap.Width(), bitmap.Height()), CV_8UC3);
                    //cout << "Convert data into mat.\n";
                    //cvImage_raw.convertTo(cvImage, CV_8UC3, 1/256.0);
                    //cv::cvtColor(cvImage_raw, cvImage, cv::COLOR_BGR2RGB);
                    ROS_INFO_STREAM("Dumping image.");
                    bool is_archiving = true;
                    if (is_archiving == true) {
                        std::string fname = dumpImage(image, bitmap, "/home/user/test.jpg");
                    }
                    toc = std::chrono::high_resolution_clock::now();
                    dt = toc - tic3;
                    ROS_INFO_STREAM("Time to compress image to jpeg and write to disk was: "
                                    << dt.count() / 1e9 << "s");

                    P1::ImageSdk::TagId ids;
                    P1::ImageSdk::ImageTag tag = image.GetTag(ids.DateTime);
                    ROS_INFO_STREAM("Tag: " << tag.ToString());

                    // Create a shared pointer for intra-process zero-copy
                    //sensor_msgs::ImagePtr img_msg(new sensor_msgs::Image);
                    //std_msgs::Header header; // empty header
                    //header.seq = counter; // user defined counter
                    //header.stamp = ros::Time::now(); // time
                    //img_bridge = cv_bridge::CvImage(header,
                    //        sensor_msgs::image_encodings::RGB8, cvImage_raw);
                    //img_bridge.toImageMsg(*img_msg);
                    //image_pub.publish(img_msg);
                    //toc = std::chrono::high_resolution_clock::now();
                    //dt = toc - tic3;
                    //ROS_INFO_STREAM("Time to compress image to jpeg and publish was: "
                    //                << dt.count() / 1e9 << "s");
                    //return 0;
                    //counter++;
                    toc = std::chrono::high_resolution_clock::now();
                    dt = toc - tic1;
                    ROS_INFO_STREAM("Time to for total call of camera trigger "
                                    "to disk: " << dt.count() / 1e9 << "s");

                }
            }
            catch (P1::ImageSdk::SdkException exception)
	        {
	        	// Exception from ImageSDK
	        	ROS_ERROR_STREAM("ImageSDK Exception: " << exception.what() << " Code:"
                                 << exception.mCode << std::endl);
	        	return -1;
	        }
	        catch (P1::CameraSdk::SdkException exception)
	        {
	        	// Exception from CameraSDK
	        	ROS_ERROR_STREAM("CameraSDK Exception: " << exception.what() << " Code:"
	        		             << exception.mErrorCode << std::endl);
			running_ = false;
	        	return -1;
	        }
	        catch (...)
	        {
	        	// Any other exception - just in case
	        	std::cout << "Argh - we got an exception" << std::endl;
	        	return -1;
	        }
            return 0;
        };

        std::string dumpImage(P1::ImageSdk::RawImage rawImage,
                              P1::ImageSdk::BitmapImage bitmap,
                              const std::string &filename)
        {
            auto start_db  = ros::Time::now();
            boost::filesystem::path path_filename{filename};
            boost::filesystem::create_directories(path_filename.parent_path());
            ROS_INFO("Writing image.");
            jpegConfig.quality = 80;
            P1::ImageSdk::JpegWriter(filename, bitmap, rawImage, jpegConfig);
            //P1::ImageSdk::TiffConfig tiffConfig;
            //P1::ImageSdk::TiffWriter(filename, bitmap, rawImage, tiffConfig);
            return filename;
        }

        void getPropertyMaps() {
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

        bool getPhaseOneParameter(phase_one::GetPhaseOneParameter::Request& req,
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
              }
              catch (const std::exception& ex)
              {
                ROS_ERROR_STREAM("Cannot get parameter: " << ex.what());
                resp.message = ex.what();
              }
            }

            return true;

        };

        bool setPhaseOneParameter(phase_one::SetPhaseOneParameter::Request& req,
                                  phase_one::SetPhaseOneParameter::Response& resp)
        {
            ROS_INFO("Received Request to Set Parameter(s).");
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
                    ROS_INFO_STREAM("Previous value: " << cam_val.ToString());
                    ROS_INFO_STREAM("name: " << name << " value: "
                                    << val << " id: " << id << "\n");
                    std::string s1 = property_to_type_[ name ].ToString();
                    if (s1.find("Enum") != std::string::npos ||
                            s1.find("Bool") != std::string::npos ||
                            s1.find("Int") != std::string::npos) {
                        P1::CameraSdk::PropertyValue pv;
                        pv.mType = property_to_type_[name].mType;
                        pv.mInt = std::stoi(val);
                        ROS_INFO_STREAM("Setting Int property: " << name <<
                                        " : to value : " << val << " :\n");
                        camera.SetProperty(id, pv);
                    } else if (s1.find("Float") != std::string::npos) {
                        P1::CameraSdk::PropertyValue pv;
                        pv.mType = property_to_type_[name].mType;
                        ROS_INFO_STREAM("Type: " << property_to_type_[name].mType << " Value: "
                                        << property_to_type_[name].ToString());
                        pv.mDouble = std::stod(val);
                        ROS_INFO_STREAM("Setting Double property: " << name <<
                                        " : to value : " << val << " :\n");
                        camera.SetProperty(id, pv);
                    } else {
                        P1::CameraSdk::PropertyValue pv;
                        pv.mType = property_to_type_[name].mType;
                        pv.mString = val;
                        ROS_INFO_STREAM("Setting String property: " << name <<
                                        " : to value : " << val << " :\n");
                        camera.SetProperty(id, pv);
                    }
                    cam_val = camera.Property(id);
                    ROS_INFO_STREAM("New value: " << cam_val.ToString());
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
                std::cout << name << std::endl;
                std::cout << value << std::endl;
                param_to_value[name] = value;
                parameters.erase(0, pos + delimiter1.length());
            }
            token = parameters;
            name = token.substr(0, token.find(delimiter2));
            token.erase(0, token.find(delimiter2) + delimiter2.length());
            value = token;
            std::cout << name << std::endl;
            std::cout << value << std::endl;
            param_to_value[name] = value;

            return param_to_value;
        };
    };

PLUGINLIB_EXPORT_CLASS(phase_one::PhaseOneNodelet, nodelet::Nodelet);
}
