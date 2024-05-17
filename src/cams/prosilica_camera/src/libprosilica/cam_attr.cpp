//
// WORK IN PROGRESS
//

#include "cam_attr.h"

class CamAttr {
public:
    CamAttr(tPvDatatype);
    std::string name = "Dummy";
    std::string dtypeName = "Dummy";
    bool set(std::string const&);
    bool get(int64_t&);
    bool get(std::string&);
    bool getStr(std::string&);
protected:

    tPvDatatype dtype;
private:
    bool (CamAttr::*setAttr)(std::string const&);
    bool (CamAttr::*getAttrVal)(std::string &);
    bool Nop_setAttr(std::string const&);
    bool Uint32_setAttr(std::string const&);
    bool Float32_setAttr(std::string const&);
    bool Int64_setAttr(std::string const&);
    bool String_setAttr(std::string const&);
    bool String_getAttr(std::string&);

};

bool CamAttr::set(std::string const &s) {
    return (this->*setAttr)(s);
}

bool CamAttr::get(std::string &s) {
    return (this->*getAttrVal)(s);
}

bool CamAttr::Nop_setAttr(std::string const& value) {
    std::cout << "D'oh! " << value << std::endl;
    return false;
}
bool CamAttr::Uint32_setAttr(std::string const& value) {
    std::cout << "called Uint32 setAttr: " << value << std::endl;
    return true;
}
bool CamAttr::Float32_setAttr(std::string const& value) {
    std::cout << "called Float32 setAttr: " << value << std::endl;
    return true;
}
bool CamAttr::Int64_setAttr(std::string const& value) {
    std::cout << "called Int64 setAttr: " << value << std::endl;
    return true;
}
bool CamAttr::String_setAttr(std::string const& value) {
    std::cout << "called String setAttr: " << value << std::endl;
    return true;
}

CamAttr::CamAttr(tPvDatatype dataType) : dtype{dataType} {
//    dtypeName = std::string(typeNames[dtype]);
//    std::cout << "new attr: " << name << " type: " << dtypeName << std::endl;
    switch (dtype) {
//        ePvDatatypeUnknown  = 0,
//        ePvDatatypeCommand  = 1,
//        ePvDatatypeRaw      = 2,
//        ePvDatatypeString   = 3,
//        ePvDatatypeEnum     = 4,
//        ePvDatatypeUint32   = 5,
//        ePvDatatypeFloat32  = 6,
//        ePvDatatypeInt64    = 7,
//        ePvDatatypeBoolean  = 8,
        case ePvDatatypeString:
            setAttr = &CamAttr::String_setAttr;
            break;
        case ePvDatatypeFloat32:
            setAttr = &CamAttr::Float32_setAttr;
            break;
        case ePvDatatypeUint32:
            setAttr = &CamAttr::Uint32_setAttr;
            break;
        case ePvDatatypeInt64:
            setAttr = &CamAttr::Int64_setAttr;
            break;
        default:
            setAttr = &CamAttr::Nop_setAttr;
    }
}

