CMD ?= bash
runtime ?= runc
network ?= host
name ?= kamera
DAQDEV ?= mcc_daq
REPO_DIR ?= $(shell realpath ${PWD})
REPO_DIR_BASEDIR ?= $(shell dirname $(REPO_DIR))
ROS_MASTER_URI ?= not_set_your_config_is_bad
## Docker buildkit - disable by setting to 0 if you have issues

DAQPATH = $(shell readlink -f "/dev/$(DAQDEV)")
PROJ_DIR=/root/noaa_kamera
WS_DIR=/root/kamera_ws
ROS_DISTRO=noetic
BRANCH=latest

.PHONY: info
info:
	@echo REPO_DIR_BASEDIR=$(REPO_DIR_BASEDIR)
	@echo ROS_MASTER_URI=$(ROS_MASTER_URI)
	@echo REDIS_HOST=$(REDIS_HOST)

.PHONY: build
build:
	docker compose build

.PHONY: build-nuvo
build-nuvo:
	ROS_DISTRO=noetic docker compose --profile nuvo build

.PHONY: build-viame
build-viame:
	ROS_DISTRO=melodic docker compose --profile viame build

.PHONY: build-gui
build-gui:
	ROS_DISTRO=kinetic docker compose --profile gui build

.PHONY: build-postflight
build-postflight:
	docker compose --profile pf build

.PHONY: clean
clean:
	rm -rf .ros .catkin_tools .cmake .config build* devel* logs*
