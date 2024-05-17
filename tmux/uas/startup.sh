#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source /opt/ros/melodic/setup.bash

rosclean purge -y
mkdir -p ~/.tmuxinator
mkdir -p ~/.config/kamera/gui
