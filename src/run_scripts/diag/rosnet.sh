#!/usr/bin/env bash

## Diagnostics for ros networking

## From satellite
# bare minimum - if this fails, you can't see the rosmaster at all
rostopic list

# Should be able to get basic info
rosnode info ${node}

