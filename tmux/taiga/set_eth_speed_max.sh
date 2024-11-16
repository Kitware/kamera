#!/usr/bin/env bash

# Set all the interface speeds to maximum
# don't do the interconnect
# this is a stupid hack and shouldn't exist but I do not understand why the
# IR camera connection is defaulting to 10M rather than 1Gbps.

for iface in 25g_outer 25g_inner 1g pci_bot pci_top; do
# this may do weird things to the IP connection of the camera ports
    ./ethtoolp.py $iface --maximize-speed
done
