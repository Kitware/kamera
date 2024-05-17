#ifndef KW_GENICAM_DRIVER_MACROS_H
#define KW_GENICAM_DRIVER_MACROS_H

// ansi color codes
#define BLU "\033[0;34m"
#define GRN "\033[0;32m"
#define RED "\033[0;31m"
#define PUR "\033[0;35m"
// No Color
#define NC "\033[0m"

// ----------------------------------------------------------------------------
// Helper Macros

/** Convert IP Address UINT32 into a string to an input stream. */
#define IP_ADDR(IP) (((IP) >> 24) & 0xff) << "." << (((IP) >> 16) & 0xff) << "." << (((IP) >> 8) & 0xff) << "." << (((IP) >> 0) & 0xff)

/** Convert MAC address low and high parts to the standard hex to an input stream.
 * << (((HIGH) >> 12) & 0xf) << (((HIGH) >>  8) & 0xf) << ":"
 */
#define MAC_ADDR(LOW, HIGH) \
  std::hex << std::setfill('0') \
      << std::setw(2) << (((HIGH) >>  8) & 0xff) << ":" \
      << std::setw(2) << (((HIGH) >>  0) & 0xff) << ":" \
      << std::setw(2) << (((LOW)  >> 24) & 0xff) << ":" \
      << std::setw(2) << (((LOW)  >> 16) & 0xff) << ":" \
      << std::setw(2) << (((LOW)  >>  8) & 0xff) << ":" \
      << std::setw(2) << (((LOW)  >>  0) & 0xff)



/**
 * Log success (info) or failure (warn) with failed status description if
 * applicable.
 */
#define WARN_ON_FAILURE(func, ret_status, success_status) \
  do \
  { \
    if( ret_status != success_status ) \
    { \
      ROS_WARN_STREAM( #func " " << decode_sdk_status( ret_status ) ); \
    } \
  } while( false )

#define WARN_ON_FAILURE_NOISY(func, ret_status, success_status) \
  do \
  { \
    if( ret_status == success_status ) \
    { \
      ROS_INFO( #func " OK" ); \
    } \
    else \
    { \
      ROS_WARN_STREAM( #func " " << decode_sdk_status( ret_status ) ); \
    } \
  } while( false )

/**
 * Log success (info) or failure (error) with failed status description if
 * applicable.  On failure, triggers a call to safe_exit with the given return
 * code and camera handle pointer.
 */
#define RETURN_ON_FAILURE(func, ret_status, success_status, failure_ret_code, handle) \
  do \
  { \
    if( ret_status == success_status ) \
    { \
      ROS_INFO( #func " OK" ); \
    } \
    else \
    { \
      ROS_ERROR_STREAM( #func " " << decode_sdk_status( ret_status ) ); \
      return safe_exit( 1, handle ); \
    } \
  } while( false )


#endif //KW_GENICAM_DRIVER_MACROS_H
