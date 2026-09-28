/* glibc >= 2.34 (Ubuntu 22.04+) removed the deprecated pthread_yield();
 * the prebuilt DALSA GigE-V libGevApi.so still references it. Provide the
 * old symbol as a thin alias for sched_yield(). */
#define _GNU_SOURCE
#include <sched.h>

int pthread_yield(void)
{
    return sched_yield();
}
