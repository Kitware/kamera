ROS_DISTRO=noetic

.PHONY: build
build:
	docker compose build

.PHONY: core
core:
	docker compose --profile core build

.PHONY: viame
viame:
	docker compose --profile viame build

.PHONY: gui
gui:
	ROS_DISTRO=kinetic docker compose --profile gui build

.PHONY: postflight
postflight:
	docker compose --profile pf build

.PHONY: clean
clean:
	rm -rf .ros .catkin_tools .cmake .config build* devel* logs*
