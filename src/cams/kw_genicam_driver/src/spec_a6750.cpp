// spec_a6750.cpp
/** Camera-specific data structures for interfacing with Genicam API */
#include <cstdio>
#include <string>
#include <stdexcept>
#include <map>
#include <cv_bridge/cv_bridge.h>

using std::string;
using std::map;


map<string, string> FirmModeToCVEnc{ {"bayer", "mono8"}, {"color", "bgr8"}, {"mono", "mono8"},
                                        {"mono8", "mono8"}, {"mono16", "mono16"}};

map<string, int> FirmModeToCVMat{ {"bayer", CV_8UC1}, {"color", CV_8UC2}, {"mono", CV_8UC1},
                              {"mono8", CV_8UC1}, {"mono16", CV_16UC1}};

enum CamModelType
{
    eCamTypeUnknown         = 0,        // unset/unknown
    eCamType6750            = 1,
    eCamType6xx             = 2,
} ;

map<string, int> CamTypeMap{ {"6750", eCamType6750},
                              {"6xx", eCamType6xx},};


enum FirmwareMode { Bayer=0, Color=1, Mono=2, Mono8=3, Mono16=4};
map<string, int> FirmModeMap{ {"bayer", Bayer}, {"color", Color}, {"mono", Mono},
                              {"mono8", Mono8}, {"mono16", Mono16}};

enum TriggerSource { TS_INTERNAL=0, TS_EXTERNAL=1, TS_SOFTWARE=2, TS_IRIG=3, TS_VIDEO=4};
map<string, int> TrigSrcMap{ {"Internal", TS_INTERNAL}, {"External", TS_EXTERNAL},
                             {"Software", TS_SOFTWARE}, {"IRIG", TS_IRIG},
                             {"Video", TS_VIDEO}};

enum IRFormat { IR_RAD=0, IR_TL100mK=1, IR_TL10mK=2};
map<string, int> IrFormatMap{ {"Radiometric", IR_RAD},
                              {"TemperatureLinear100mK", IR_TL100mK},
                              {"TemperatureLinear10mK", IR_TL10mK}
};