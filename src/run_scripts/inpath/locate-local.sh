#!/usr/bin/env bash

mkdir -p ~/.config
rm -f ~/.config/localcam.txt

# Locate link-local devices
ping -c 1 -b 169.254.255.255 2>/dev/null |\
  grep -P "(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b)" -o |\
  grep -P "(169\.254\.\d{1,3}\.\d{1,3}\b)" |\
  grep -v 255 | grep -vP "^172" > ~/.config/linklocal.txt

for addr in `cat ~/.config/linklocal.txt`; do
    ping -c 1 $addr 2>/dev/null |\
        grep -P "(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b)" -o |\
            grep -v 255 >> ~/.config/localcam.txt; done

cat ~/.config/localcam.txt | uniq