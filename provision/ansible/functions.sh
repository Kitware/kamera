function kamera_uas_provision() {
	ansible-playbook playbooks/provision.yml -i hosts.yml --ask-become-pass -u "user" -e 'ansible_python_interpreter=/usr/bin/python3' --limit $1
}

function kamera_uas_configure() {
	ansible-playbook playbooks/configure.yml -i hosts.yml --ask-become-pass --tags "uas" --ask-become-pass -u "user" -e 'ansible_python_interpreter=/usr/bin/python3' --limit $1
}

function kamera_uas_build() {
	ansible-playbook playbooks/build.yml -i hosts.yml -u "user" -e 'ansible_python_interpreter=/usr/bin/python3' --limit $1
}

function kamera_uas_all() {
	ansible-playbook playbooks/all.yml -i hosts.yml --ask-become-pass -u "user" -e 'ansible_python_interpreter=/usr/bin/python3' --limit $1
}

function kamera_uas_synchronize() {
	ansible-playbook playbooks/sync.yml -i hosts.yml --ask-become-pass -u "user" -e 'ansible_python_interpreter=/usr/bin/python3' --limit $1
}

function kamera_cas_test() {
	ansible-playbook playbooks/cas/test.yml -i hosts.yml --ask-become-pass -u "user" -e 'ansible_python_interpreter=/usr/bin/python3' --limit $1
}

function kamera_cas_configure() {
	ansible-playbook playbooks/cas/configure.yml -i hosts.yml --ask-become-pass -u "user" -e 'ansible_python_interpreter=/usr/bin/python3' --limit $1
}

function kamera_cas_provision() {
	ansible-playbook playbooks/cas/provision.yml -i hosts.yml --ask-become-pass -u "user" -e 'ansible_python_interpreter=/usr/bin/python3' --limit $1
}

function kamera_cas_build() {
	ansible-playbook playbooks/cas/build.yml -i hosts.yml --ask-become-pass -u "user" -e 'ansible_python_interpreter=/usr/bin/python3' --limit $1
}

function kamera_cas_all() {
	ansible-playbook playbooks/cas/all.yml -i hosts.yml --ask-become-pass -u "user" -e 'ansible_python_interpreter=/usr/bin/python3' --limit $1
}
