#!/bin/bash

echo $PATH | awk -v RS=: -v ORS=: '!($0 in a) {a[$0]; print}'