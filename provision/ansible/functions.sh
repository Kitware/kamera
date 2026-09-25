function kamera_configure() {
	ansible-playbook playbooks/configure.yml -i hosts.yml --ask-become-pass -u "user" -e 'ansible_python_interpreter=/usr/bin/python3' --limit $1
}

function kamera_provision() {
	ansible-playbook playbooks/provision.yml -i hosts.yml --ask-become-pass -u "user" -e 'ansible_python_interpreter=/usr/bin/python3' --limit $1
}

function kamera_build() {
	ansible-playbook playbooks/build.yml -i hosts.yml --ask-become-pass -u "user" -e 'ansible_python_interpreter=/usr/bin/python3' --limit $1
}

function kamera_all() {
	ansible-playbook playbooks/all.yml -i hosts.yml --ask-become-pass -u "user" -e 'ansible_python_interpreter=/usr/bin/python3' --limit $1
}

function kamera_host_for_position() {
	local position=$1
	local system_name
	system_name=$(tr -d '[:space:]' < /home/user/kw/SYSTEM_NAME)
	case $position in
		center) echo "center0${system_name}" ;;
		left)   echo "left1${system_name}" ;;
		right)  echo "right2${system_name}" ;;
		*)
			echo "Unknown position: ${position} (expected center, left, or right)" >&2
			return 1
			;;
	esac
}
