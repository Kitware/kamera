/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "decode_error.h"

#include <sstream>
#include <map>

//=========================================================================
// GevLib error codes.
//=========================================================================

#define TABLE_ENTRY(C,T) { C, #C " - " T },

static std::map< GEV_STATUS, const char* > sdk_error_codes = {

  TABLE_ENTRY( GEVLIB_OK,                                "Success" )
  TABLE_ENTRY( GEVLIB_SUCCESS,                           "Success" )
  TABLE_ENTRY( GEVLIB_STATUS_SUCCESS,                    "Success" )
  TABLE_ENTRY( GEVLIB_STATUS_ERROR,                      "Generic Error. A catch-all for unexpected behaviour" )

//standard api errors
  TABLE_ENTRY( GEVLIB_ERROR_GENERIC,                     "Generic Error. A catch-all for unexpected behaviour" )
  TABLE_ENTRY( GEVLIB_ERROR_NULL_PTR,                    "NULL pointer passed to function or the result of a cast operation" )
  TABLE_ENTRY( GEVLIB_ERROR_ARG_INVALID,                 "Passed argument to a function is not valid" )
  TABLE_ENTRY( GEVLIB_ERROR_INVALID_HANDLE,              "Invalid Handle" )
  TABLE_ENTRY( GEVLIB_ERROR_NOT_SUPPORTED,               "This version of hardware/fpga does not support this feature" )
  TABLE_ENTRY( GEVLIB_ERROR_TIME_OUT,                    "Timed out waiting for a resource" )
  TABLE_ENTRY( GEVLIB_ERROR_NOT_IMPLEMENTED,             "Function / feature is not implemented." )
  TABLE_ENTRY( GEVLIB_ERROR_NO_CAMERA,                   "The action can't be execute because the camera is not connected." )
  TABLE_ENTRY( GEVLIB_ERROR_INVALID_PIXEL_FORMAT,        "Pixel Format is invalid (not supported or not recognized)" )
  TABLE_ENTRY( GEVLIB_ERROR_PARAMETER_INVALID,           "Passed Parameter (could be inside a data structure) is invalid/out of range." )
  TABLE_ENTRY( GEVLIB_ERROR_SOFTWARE,                    "software error, unexpected result" )
  TABLE_ENTRY( GEVLIB_ERROR_API_NOT_INITIALIZED,         "API has not been initialized" )
  TABLE_ENTRY( GEVLIB_ERROR_DEVICE_NOT_FOUND,            "Device/camera specified was not found." )
  TABLE_ENTRY( GEVLIB_ERROR_ACCESS_DENIED,               "API will not access the device/camera/feature in the specified manner." )
  TABLE_ENTRY( GEVLIB_ERROR_NOT_AVAILABLE,               "Feature / function is not available for access (but is implemented)." )
  TABLE_ENTRY( GEVLIB_ERROR_NO_SPACE,                    "The data being written to a feature is too large for the feature to store." )

// Resource errors.
  TABLE_ENTRY( GEVLIB_ERROR_SYSTEM_RESOURCE,             "Error creating a system resource" )
  TABLE_ENTRY( GEVLIB_ERROR_INSUFFICIENT_MEMORY,         "error allocating memory" )
  TABLE_ENTRY( GEVLIB_ERROR_INSUFFICIENT_BANDWIDTH,      "Not enough bandwidth to perform operation and/or acquisition" )
  TABLE_ENTRY( GEVLIB_ERROR_RESOURCE_NOT_ALLOCATED,      "Resource is not currently allocated" )
  TABLE_ENTRY( GEVLIB_ERROR_RESOURCE_IN_USE,             "resource is currently being used." )
  TABLE_ENTRY( GEVLIB_ERROR_RESOURCE_NOT_ENABLED,        "The resource(feature) is not enabled" )
  TABLE_ENTRY( GEVLIB_ERROR_RESOURCE_NOT_INITIALIZED,    "Resource has not been intialized." )
  TABLE_ENTRY( GEVLIB_ERROR_RESOURCE_CORRUPTED,          "Resource has been corrupted" )
  TABLE_ENTRY( GEVLIB_ERROR_RESOURCE_MISSING,            "A resource (ie.DLL) needed could not located" )
  TABLE_ENTRY( GEVLIB_ERROR_RESOURCE_LACK,               "Lack of resource to perform a request." )
  TABLE_ENTRY( GEVLIB_ERROR_RESOURCE_ACCESS,             "Unable to correctly access the resource." )
  TABLE_ENTRY( GEVLIB_ERROR_RESOURCE_INVALID,            "A specified resource does not exist." )
  TABLE_ENTRY( GEVLIB_ERROR_RESOURCE_LOCK,               "resource is currently lock" )
  TABLE_ENTRY( GEVLIB_ERROR_INSUFFICIENT_PRIVILEGE,      "Need administrator privilege." )
  TABLE_ENTRY( GEVLIB_ERROR_RESOURCE_WRITE_PROTECTED,    "No data can be written to the resource" )
  TABLE_ENTRY( GEVLIB_ERROR_RESOURCE_INCOHERENCY,        "The required resources are not valid together" )

// Data errors
  TABLE_ENTRY( GEVLIB_ERROR_DATA_NO_MESSAGES,            "no more messages (in fifo, queue, input stream etc)" )
  TABLE_ENTRY( GEVLIB_ERROR_DATA_OVERFLOW,               "data could not be added to fifo, queue, stream etc." )
  TABLE_ENTRY( GEVLIB_ERROR_DATA_CHECKSUM,               "checksum validation fail" )
  TABLE_ENTRY( GEVLIB_ERROR_DATA_NOT_AVAILABLE,          "data requested isn't available yet" )
  TABLE_ENTRY( GEVLIB_ERROR_DATA_OVERRUN,                "data requested has been overrun by newer data" )
  TABLE_ENTRY( GEVLIB_ERROR_DATA_XFER_ABORT,             "transfer of requested data did not finish" )
  TABLE_ENTRY( GEVLIB_ERROR_DATA_INVALID_HEADER,         "data header is invalid." )
  TABLE_ENTRY( GEVLIB_ERROR_DATA_ALIGNMENT,              "data is not correctly align." )

// Ethernet errors
  TABLE_ENTRY( GEVLIB_ERROR_CONNECTION_DROPPED,          "NO DESCRIPTION" )
  TABLE_ENTRY( GEVLIB_ERROR_ANSWER_TIMEOUT,              "NO DESCRIPTION" )
  TABLE_ENTRY( GEVLIB_ERROR_SOCKET_INVALID,              "NO DESCRIPTION" )
  TABLE_ENTRY( GEVLIB_ERROR_PORT_NOT_AVAILABLE,          "NO DESCRIPTION" )
  TABLE_ENTRY( GEVLIB_ERROR_INVALID_IP,                  "NO DESCRIPTION" )
  TABLE_ENTRY( GEVLIB_ERROR_INVALID_CAMERA_OPERATION,    "NO DESCRIPTION" )
  TABLE_ENTRY( GEVLIB_ERROR_INVALID_PACKET,              "NO DESCRIPTION" )
  TABLE_ENTRY( GEVLIB_ERROR_INVALID_CONNECTION_ATTEMPT,  "NO DESCRIPTION" )
  TABLE_ENTRY( GEVLIB_ERROR_PROTOCOL,                    "NO DESCRIPTION" )
  TABLE_ENTRY( GEVLIB_ERROR_WINDOWS_SOCKET_INIT,         "NO DESCRIPTION" )
  TABLE_ENTRY( GEVLIB_ERROR_WINDOWS_SOCKET_CLOSE,        "NO DESCRIPTION" )
  TABLE_ENTRY( GEVLIB_ERROR_SOCKET_CREATE,               "NO DESCRIPTION" )
  TABLE_ENTRY( GEVLIB_ERROR_SOCKET_RELEASE,              "NO DESCRIPTION" )
  TABLE_ENTRY( GEVLIB_ERROR_SOCKET_DATA_SEND,            "NO DESCRIPTION" )
  TABLE_ENTRY( GEVLIB_ERROR_SOCKET_DATA_READ,            "NO DESCRIPTION" )
  TABLE_ENTRY( GEVLIB_ERROR_SOCKET_WAIT_ACKNOWLEDGE,     "NO DESCRIPTION" )
  TABLE_ENTRY( GEVLIB_ERROR_INVALID_INTERNAL_COMMAND,    "NO DESCRIPTION" )
  TABLE_ENTRY( GEVLIB_ERROR_INVALID_ACKNOWLEDGE,         "NO DESCRIPTION" )
  TABLE_ENTRY( GEVLIB_ERROR_PREVIOUS_ACKNOWLEDGE,        "NO DESCRIPTION" )
  TABLE_ENTRY( GEVLIB_ERROR_INVALID_MESSAGE,             "NO DESCRIPTION" )
  TABLE_ENTRY( GEVLIB_ERROR_GIGE_ERROR,                  "NO DESCRIPTION" )

  // ----

  TABLE_ENTRY( GEV_STATUS_SUCCESS,             "Requested operation was completed successfully." )
  TABLE_ENTRY( GEV_STATUS_NOT_IMPLEMENTED,     "The request isn't supported by the device." )
  TABLE_ENTRY( GEV_STATUS_INVALID_PARAMETER,   "At least one parameter provided in the command is invalid (or out of range) for the device" )
  TABLE_ENTRY( GEV_STATUS_INVALID_ADDRESS,     "An attempt was made to access a non existent address space location." )
  TABLE_ENTRY( GEV_STATUS_WRITE_PROTECT,       "The addressed register must not be written." )
  TABLE_ENTRY( GEV_STATUS_BAD_ALIGNMENT,       "A badly aligned address offset or data size was specified." )
  TABLE_ENTRY( GEV_STATUS_ACCESS_DENIED,       "An attempt was made to access an address location which is currently/momentary not accessible." )
  TABLE_ENTRY( GEV_STATUS_BUSY,                "A required resource to service the request isn't currently available. The request may be retried." )
  TABLE_ENTRY( GEV_STATUS_LOCAL_PROBLEM,       "A internal problem in the device implementation occurred while processing the request." )
  TABLE_ENTRY( GEV_STATUS_MSG_MISMATCH,        "Message mismatch (request and acknowledge don't match)" )
  TABLE_ENTRY( GEV_STATUS_INVALID_PROTOCOL,    "This version of the GVCP protocol is not supported" )
  TABLE_ENTRY( GEV_STATUS_NO_MSG,              "No message received, timeout." )
  TABLE_ENTRY( GEV_STATUS_PACKET_UNAVAILABLE,  "The request packet is not available anymore." )
  TABLE_ENTRY( GEV_STATUS_DATA_OVERRUN,        "Internal memory of device overrun (typically for image acquisition)" )
  TABLE_ENTRY( GEV_STATUS_INVALID_HEADER,      "The message header is not valid. Some of its fields do not match the specificiation." )

  TABLE_ENTRY( GEV_STATUS_ERROR,               "Generic error." )

  // ----

  TABLE_ENTRY( GEV_FRAME_STATUS_RECVD,      "Frame is complete.")
  TABLE_ENTRY( GEV_FRAME_STATUS_PENDING,    "Frame is not ready.")
  TABLE_ENTRY( GEV_FRAME_STATUS_TIMEOUT,    "Frame was not ready before timeout condition met.")
  TABLE_ENTRY( GEV_FRAME_STATUS_OVERFLOW,   "Frame was not complete before the max number of frames to buffer queue was full.")
  TABLE_ENTRY( GEV_FRAME_STATUS_BANDWIDTH,  "Frame had too many resend operations due to insufficient bandwidth.")
  TABLE_ENTRY( GEV_FRAME_STATUS_LOST,       "Frame had resend operations that failed.")
  TABLE_ENTRY( GEV_FRAME_STATUS_RELEASED,   "(Internal) Frame has been released for re-use.")

};


// ----------------------------------------------------------------------------
std::string decode_sdk_status( GEV_STATUS code )
{
  if ( sdk_error_codes.count(code) > 0)
  {
    return sdk_error_codes[code];
  }

  std::stringstream msg;
  msg << "Unknown sdk error code - " << code;
  return msg.str();
}


// ============================================================================
static std::map< int, const char* > sdk_image_format = {

  TABLE_ENTRY( fmtMono8,                 "8 Bit Monochrome Unsigned" )
  TABLE_ENTRY( fmtMono8Signed,           "8 Bit Monochrome Signed" )
  TABLE_ENTRY( fmtMono10,                "10 Bit Monochrome Unsigned" )
  TABLE_ENTRY( fmtMono10Packed,          "10 Bit Monochrome Packed" )
  TABLE_ENTRY( fmtMono12,                "12 Bit Monochrome Unsigned" )
  TABLE_ENTRY( fmtMono12Packed,          "12 Bit Monochrome Packed" )
  TABLE_ENTRY( fmtMono14,                "14 Bit Monochrome Unsigned" )
  TABLE_ENTRY( fmtMono16,                "16 Bit Monochrome Unsigned" )
  TABLE_ENTRY( fMtBayerGR8,              "8-bit Bayer" )
  TABLE_ENTRY( fMtBayerRG8,              "8-bit Bayer" )
  TABLE_ENTRY( fMtBayerGB8,              "8-bit Bayer" )
  TABLE_ENTRY( fMtBayerBG8,              "8-bit Bayer" )
  TABLE_ENTRY( fMtBayerGR10,             "10-bit Bayer" )
  TABLE_ENTRY( fMtBayerRG10,             "10-bit Bayer" )
  TABLE_ENTRY( fMtBayerGB10,             "10-bit Bayer" )
  TABLE_ENTRY( fMtBayerBG10,             "10-bit Bayer" )
  TABLE_ENTRY( fMtBayerGR12,             "12-bit Bayer" )
  TABLE_ENTRY( fMtBayerRG12,             "12-bit Bayer" )
  TABLE_ENTRY( fMtBayerGB12,             "12-bit Bayer" )
  TABLE_ENTRY( fMtBayerBG12,             "12-bit Bayer" )
  TABLE_ENTRY( fmtRGB8Packed,            "8 Bit RGB Unsigned in 24bits" )
  TABLE_ENTRY( fmtBGR8Packed,            "8 Bit BGR Unsigned in 24bits" )
  TABLE_ENTRY( fmtRGBA8Packed,           "8 Bit RGB Unsigned" )
  TABLE_ENTRY( fmtBGRA8Packed,           "8 Bit BGR Unsigned" )
  TABLE_ENTRY( fmtRGB10Packed,           "10 Bit RGB Unsigned" )
  TABLE_ENTRY( fmtBGR10Packed,           "10 Bit BGR Unsigned" )
  TABLE_ENTRY( fmtRGB12Packed,           "12 Bit RGB Unsigned" )
  TABLE_ENTRY( fmtBGR12Packed,           "12 Bit BGR Unsigned" )
  TABLE_ENTRY( fmtRGB10V1Packed,         "10 Bit RGB custom V1 (32bits)*/" )
  TABLE_ENTRY( fmtRGB10V2Packed,         "10 Bit RGB custom V2 (32bits)*/" )
  TABLE_ENTRY( fmtYUV411packed,          "YUV411 (composite color)" )
  TABLE_ENTRY( fmtYUV422packed,          "YUV422 (composite color)" )
  TABLE_ENTRY( fmtYUV444packed,          "YUV444 (composite color)" )
  TABLE_ENTRY( fmt_PFNC_YUV422_8,        "YUV 4:2:2 8-bit" )
  TABLE_ENTRY( fmtRGB8Planar,            "RGB8 Planar buffers" )
  TABLE_ENTRY( fmtRGB10Planar,           "RGB10 Planar buffers" )
  TABLE_ENTRY( fmtRGB12Planar,           "RGB12 Planar buffers" )
  TABLE_ENTRY( fmtRGB16Planar,           "RGB16 Planar buffers" )
  TABLE_ENTRY( fmt_PFNC_BiColorBGRG8,    "Bi-color Blue/Green - Red/Green 8-bit" )
  TABLE_ENTRY( fmt_PFNC_BiColorBGRG10,   "Bi-color Blue/Green - Red/Green 10-bit unpacked" )
  TABLE_ENTRY( fmt_PFNC_BiColorBGRG10p,  "Bi-color Blue/Green - Red/Green 10-bit packed" )
  TABLE_ENTRY( fmt_PFNC_BiColorBGRG12,   "Bi-color Blue/Green - Red/Green 12-bit unpacked" )
  TABLE_ENTRY( fmt_PFNC_BiColorBGRG12p,  "Bi-color Blue/Green - Red/Green 12-bit packed" )
  TABLE_ENTRY( fmt_PFNC_BiColorRGBG8,    "Bi-color Red/Green - Blue/Green 8-bit" )
  TABLE_ENTRY( fmt_PFNC_BiColorRGBG10,   "Bi-color Red/Green - Blue/Green 10-bit unpacked" )
  TABLE_ENTRY( fmt_PFNC_BiColorRGBG10p,  "Bi-color Red/Green - Blue/Green 10-bit packed" )
  TABLE_ENTRY( fmt_PFNC_BiColorRGBG12,   "Bi-color Red/Green - Blue/Green 12-bit unpacked" )
  TABLE_ENTRY( fmt_PFNC_BiColorRGBG12p,  "Bi-color Red/Green - Blue/Green 12-bit packed" )

};


// ----------------------------------------------------------------------------
const char* decode_pixel_format( int code )
{
  if ( sdk_image_format.count(code) > 0 )
  {
    return sdk_image_format[code];
  }

  std::stringstream msg;
  msg << "Unknown pixel format code - " << code;
  return msg.str().c_str();
}


#undef TABLE_ENTRY
