#!/usr/bin/env bash

# Set all the interface speeds to maximum
# don't do the interconnect
# this is a stupid hack and shouldn't exist but I do not understand why the
# IR camera connection is defaulting to 10M rather than 1Gbps.

for iface in mobo_top mobo_btm 1g_pci_bot pci_btm pci_top; do
# this may do weird things to the IP connection of the camera ports
    /home/user/kw/kamera/tmux/taiga/ethtoolp.py $iface --maximize-speed
done
