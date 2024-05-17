// cam_spec.h
/* Configuration Parameters to match genicam driver with specifics of the device's GigEV XML-defined interface */

#ifndef KW_GENICAM_DRIVER_CAM_SPEC_H
#define KW_GENICAM_DRIVER_CAM_SPEC_H

#include <cstdio>
#include <string>
#include <stdexcept>
#include <map>

// Definitions taken from genicam example driver. Assuming OK.
#define MAX_NETIF             8
#define MAX_CAMERAS_PER_NETIF 32
#define MAX_CAMERAS           (MAX_NETIF * MAX_CAMERAS_PER_NETIF)

// ----------------------------------------------------------------------------

extern std::map<std::string, std::string> FirmModeToCVEnc;

extern std::map<std::string, int> FirmModeToCVMat;

extern std::map<std::string, int> CamTypeMap;

extern std::map<std::string, int> FirmModeMap;

extern std::map<std::string, int> TrigSrcMap;

extern std::map<std::string, int> IrFormatMap;

// ----------------------------------------------------------------------------

#endif //KW_GENICAM_DRIVER_CAM_SPEC_H
