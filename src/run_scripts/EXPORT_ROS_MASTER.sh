. "$(catkin locate)/activate_ros.bash"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROS_MASTER_URI="$(cat "${SCRIPT_DIR}/ROS_MASTER.txt")"
