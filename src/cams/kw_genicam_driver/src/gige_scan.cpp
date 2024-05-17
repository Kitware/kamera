#include <cstring>
#include <string>
#include <iomanip>

// Includes from: /opt/genicam_v3_0/library/CPP/include/
#include <GenApi/GenApi.h>
// Includes from: /usr/dalsa/GigeV/include/
#include <gevapi.h>

// ----------------------------------------------------------------------------
#define MAX_NETIF             8
#define MAX_CAMERAS_PER_NETIF 32
#define MAX_CAMERAS           (MAX_NETIF * MAX_CAMERAS_PER_NETIF)

void hexDump(const char *desc, const void *addr, const int len, uint8_t offset) {
    if (offset > 32) {
        fprintf(stderr, "failed to print buffer, offset too big\n");
        return;
    }
    int i;
    unsigned char buff[33];
    const unsigned char *pc = (const unsigned char *) addr;

    // Output description if given.
    if (desc != NULL) fprintf(stderr, "%s:\n", desc);

    if (len <= 0) {
        fprintf(stderr, "  ZERO LENGTH\n");
        return;
    }

    // Process every byte in the data.
    for (i = 0; i < len; i++) {
        // Multiple of 16 means new line (with line offset).
        if ((i % offset) == 0) {

            // Just don't print ASCII for the zeroth line.
            if (i != 0) {
                fprintf(stderr, "  %s", buff);
                fprintf(stderr, "\n");
            }
            // Output the offset.
            fprintf(stderr, "  %04x ", i);
        }
        // Now the hex code for the specific character.
        fprintf(stderr, " %02x", pc[i]);

        // And store a printable ASCII character for later.
        if ((pc[i] < 0x20) || (pc[i] > 0x7e)) {
            buff[i % offset] = '.';
        } else {
            buff[i % offset] = pc[i];
        }
        buff[(i % offset) + 1] = '\0';
    }

    // Pad out last line if not exactly 16 characters.
    while ((i % offset) != 0) {
        fprintf(stderr, "   ");
        i++;
    }
    // And print the final ASCII bit.
    fprintf(stderr, "  %s\n", buff);
}

#define IP_ADDR(IP) (((IP) >> 24) & 0xff) << "." << (((IP) >> 16) & 0xff) << "." << (((IP) >> 8) & 0xff) << "." << (((IP) >> 0) & 0xff)

#define MAC_ADDR(LOW, HIGH) \
  std::hex << std::setfill('0') \
      << std::setw(2) << (((HIGH) >>  8) & 0xff) << ":" \
      << std::setw(2) << (((HIGH) >>  0) & 0xff) << ":" \
      << std::setw(2) << (((LOW)  >> 24) & 0xff) << ":" \
      << std::setw(2) << (((LOW)  >> 16) & 0xff) << ":" \
      << std::setw(2) << (((LOW)  >>  8) & 0xff) << ":" \
      << std::setw(2) << (((LOW)  >>  0) & 0xff)

std::string to_json_tuple(const std::string &ss_key, const std::string &ss_val) {
    std::stringstream out_ss;
    std::string q = "\"";
    out_ss << q << ss_key << q << ": " \
        << q << ss_val << q;
    return out_ss.str();
}

std::string to_json_tuple(const char *key, const std::string &ss_val) {
    return to_json_tuple(std::string(key), ss_val);
}

void log_camera_interface(GEV_DEVICE_INTERFACE const &itf) {
    std::stringstream tmp_ip;
    std::stringstream tmp_mac;
    std::stringstream out;
    std::string l1 = "  ";
    std::string l2 = "    ";
    tmp_ip << IP_ADDR(itf.ipAddr);
    tmp_mac <<  MAC_ADDR(itf.macLow, itf.macHigh);

    // key
    out << "\n" << l1 << "\"" << tmp_mac.str() << "\"" << ": {\n";

    // object
    out << l2 << to_json_tuple("ip", tmp_ip.str()) << ",\n";
    out << l2 << to_json_tuple("mac", tmp_mac.str()) << ",\n";
    out << l2 << to_json_tuple("manufacturer", itf.manufacturer) << ",\n";
    out << l2 << to_json_tuple("model", itf.model) << ",\n";
    out << l2 << to_json_tuple("serial", itf.serial) << ",\n";
    out << l2 << to_json_tuple("version", itf.version) << ",\n";
    out << l2 << to_json_tuple("username", itf.username) << "\n";
    out << l1 << "}";
    std::cout << out.str();

}

void log_camera_interfaces(GEV_DEVICE_INTERFACE const *interfaces, int const &num_interfaces) {
    GEV_DEVICE_INTERFACE const *itf;
    printf("{");
    for (int i = 0; i < num_interfaces; ++i) {
        // See gevapi.h for struct definition.
        itf = &interfaces[i];
        log_camera_interface(*itf);
        if (i < num_interfaces - 1) {
            printf(",");
        }
    }
    printf("\n}\n");
}

void dump_config_struct(GEVLIB_CONFIG_OPTIONS *options) {
    for (int i = 0; i < sizeof(GEVLIB_CONFIG_OPTIONS); i++) {
    }
}

// ----------------------------------------------------------------------------
int main(int argc, char **argv) {
    for (int i = 0; i < argc; i++ ){
        fprintf(stderr, "%d: %s\n", i, argv[i]);
    }
    int c;

    GEVLIB_CONFIG_OPTIONS options = {};
    GevGetLibraryConfigOptions( &options);

    while ((c = getopt (argc, argv, "t:h?")) != -1) {
        switch (c) {
            case 't':
                if (optarg) {
                    long timeout_l = atol(optarg);
                    options.discovery_timeout_ms = (UINT32) timeout_l;
                    fprintf(stderr, "timeout: %u\n", options.discovery_timeout_ms);
                }
                break;
            case '?':
            case 'h':
            {
                fprintf(stderr, "usage goes here \n");
                break;
            }
            default:
                break;
        }
    }

    hexDump("options", &options, sizeof(GEVLIB_CONFIG_OPTIONS), 4);
    GevSetLibraryConfigOptions( &options);

    GEV_DEVICE_INTERFACE camera_interfaces[MAX_CAMERAS_PER_NETIF];
    int num_interfaces(0);
    GevGetCameraList(camera_interfaces, MAX_CAMERAS, &num_interfaces);
    log_camera_interfaces(camera_interfaces, num_interfaces);
    /*
    // test code
    GEV_DEVICE_INTERFACE iface{
        0, 3232292631, 3232235520,57111,
        0x987654, 0x123456,
        {0, 0, 0, 0, 0},
        0, 0, "rockwell", "turboencabulator",
        "deadbeef9000", "5000", "bob"

    };
    camera_interfaces[0] = iface;
    camera_interfaces[1] = iface;
    camera_interfaces[1].macHigh = 0x563412;
    camera_interfaces[2] = iface;
    camera_interfaces[2].macHigh = 0x654321;
    log_camera_interfaces(camera_interfaces, 3);
*/
}
