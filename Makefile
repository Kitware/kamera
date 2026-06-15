ROS_DISTRO ?= noetic

.PHONY: build core viame gui postflight follower leader all clean

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
	docker compose --profile gui build core-ros
	docker compose --profile gui build core-deps
	docker compose --profile gui build gui-deps
	docker compose --profile gui build gui

postflight:
	docker compose --profile pf build

## --- deployment-target targets ---

# Follower node: headless sensor node — core + VIAME + postproc
follower:
	docker compose --profile follower build core-ros
	docker compose --profile follower build core-deps
	docker compose --profile follower build viame-base
	docker compose --profile follower build

# Leader node: operator workstation — GUI + postproc
leader:
	docker compose --profile leader build core-ros
	docker compose --profile leader build core-deps
	docker compose --profile leader build gui-deps
	docker compose --profile leader build

# Everything
all:
	docker compose --profile all build core-ros
	docker compose --profile all build core-deps
	docker compose --profile all build viame-base
	docker compose --profile all build gui-deps
	docker compose --profile all build

clean:
	rm -rf .ros .catkin_tools .cmake .config build* devel* logs*
