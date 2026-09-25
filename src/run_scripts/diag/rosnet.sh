#!/usr/bin/env bash

## Diagnostics for ros networking

## From satellite
# bare minimum - if this fails, DDS discovery is broken
ros2 topic list

# Should be able to get basic info
ros2 node info ${node}
