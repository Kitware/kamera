#!/bin/bash

# Source environment variables for the project.sh entrypoint. No exec.

echo "~~~ ~~~ Project entry point  project_env.sh ~~~ ~~~"
for VNAME in ROS_DISTRO REPO_DIR
do
  if [[ -z "${!VNAME}" || "${!VNAME}" == 'null' ]]; then
    printf "<project_env.sh!> Unable to determine $VNAME. Check project_env.sh and user-config\n"
    exit 1
  fi
done

checkPath () {
        case ":$PATH:" in
                *":$1:"*) return 1
                        ;;
        esac
        return 0;
}

# Prepend to $PATH
prependToPath () {
        for a; do
                checkPath $a
                if [ $? -eq 0 ]; then
                        PATH=$a:$PATH
                fi
        done
        export PATH
}

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "${REPO_DIR}/scripts/activate_ros.bash"
export REDIS_HOST=${REDIS_HOST}
export ARCH_KEY="/sys/arch/"
# Try turning respawning back to roslaunch
export NORESPAWN="true"

prependToPath $HOME/.local/bin
prependToPath $REPO_DIR/src/run_scripts/inpath
prependToPath /opt/ros/$ROS_DISTRO/bin

errcho() {
    (>&2 echo -e "\e[31m$1\e[0m")
}

export KAM_REPO_DIR=/root/kamera
