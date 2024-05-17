#include "aux.h"

void msleep(uint32_t milliseconds) {
    usleep(milliseconds * 1000);
}