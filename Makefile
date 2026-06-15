ROS_DISTRO ?= noetic

.PHONY: build core viame gui gui-noetic postflight follower leader all clean

build:
	docker compose build

## --- individual layer targets ---

core:
	docker compose --profile core build core-ros
	docker compose --profile core build core-deps
	docker compose --profile core build

viame:
	docker compose --profile viame build viame-base
	docker compose --profile viame build

gui:
	ROS_DISTRO=kinetic docker compose --profile gui build core-deps-kinetic
	ROS_DISTRO=kinetic docker compose --profile gui build core-gui-deps
	ROS_DISTRO=kinetic docker compose --profile gui build

gui-noetic:
	docker compose --profile gui-noetic build core-ros
	docker compose --profile gui-noetic build core-deps
	docker compose --profile gui-noetic build gui-deps-noetic
	docker compose --profile gui-noetic build gui-noetic

postflight:
	docker compose --profile pf build

## --- deployment-target targets ---

# Follower node: headless sensor node — core + VIAME + postproc
follower:
	docker compose --profile follower build core-ros
	docker compose --profile follower build core-deps
	docker compose --profile follower build viame-base
	docker compose --profile follower build

# Leader node: operator workstation — Kinetic GUI + postproc
leader:
	ROS_DISTRO=kinetic docker compose --profile leader build core-deps-kinetic
	ROS_DISTRO=kinetic docker compose --profile leader build core-gui-deps
	ROS_DISTRO=kinetic docker compose --profile leader build

# Everything
all:
	docker compose --profile all build core-ros
	docker compose --profile all build core-deps
	docker compose --profile all build viame-base
	docker compose --profile all build core-deps-kinetic
	docker compose --profile all build core-gui-deps
	docker compose --profile all build

clean:
	rm -rf .ros .catkin_tools .cmake .config build* devel* logs*
