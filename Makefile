ROS_DISTRO ?= noetic

# Disable docker compose 'bake' so build order follows depends_on statements
COMPOSE_BAKE = false
export COMPOSE_BAKE

.PHONY: build core viame gui gui-noetic postflight follower leader all clean

build:
	docker compose build

## --- individual layer targets ---

core:
	docker compose --profile core build

viame:
	docker compose --profile viame build

gui:
	ROS_DISTRO=kinetic docker compose --profile gui build

gui-noetic:
	docker compose --profile gui-noetic build

postflight:
	docker compose --profile pf build

## --- deployment-target targets ---

# Follower node: headless sensor node — core + VIAME + postproc
follower:
	docker compose --profile follower build

# Leader node: operator workstation — Kinetic GUI + postproc
leader:
	ROS_DISTRO=kinetic docker compose --profile leader build

# Everything
all:
	docker compose --profile all build

clean:
	rm -rf .ros .catkin_tools .cmake .config build* devel* logs*
