

#ifndef GEVLIB_DECODE_ERROR_H
#define GEVLIB_DECODE_ERROR_H

#include <string>
#include <gevapi.h>

std::string decode_sdk_status( GEV_STATUS code );
const char* decode_pixel_format( int code );

#endif // GEVLIB_DECODE_ERROR_H
