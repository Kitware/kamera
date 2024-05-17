#!/bin/bash

echo "Starting events spoofing..."
echo "WARNING: Must restart stack to turn off spoofing."
python /home/user/kw/noaa_kamera/system.py nuvo0 restart spoof
