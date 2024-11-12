#!/usr/bin/env bash

## Some basic quality of life aliases
alias d="docker"
alias doc="docker compose"
alias dps="docker-pretty-ps 2>/dev/null || docker ps"
alias dna="docker network ls"
alias e="echo"
alias ga.="git add . && git status"
alias gcm="git commit -m"
alias gst="git status"
alias h="history -i | grep -Pv '^ *[0-9]+  [[:digit:]\:\- ]{16}  h '| grep "
alias gp="git pull --ff-only"
alias plsub="git pull --recurse-submodules && git submodule update --init --recursive"
alias p="ping -c 1"
alias pros="ping -c 1 $ROS_MASTER_URI"
alias sapt="sudo apt-get install"
alias ..="cd .."
alias ll="ls -lh --color=tty"
alias getipy="pip install ipython==5 ipdb"
alias wat="hostname && who am i"
alias hosts="cat /etc/hosts"
alias resolv="cat /etc/resolv.conf"
alias ips="ip -br addr"
alias lsip="ip addr | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*'"

alias dfs="df -h | grep -v loop"

alias kamlauncher="bash ~/kw/kamera/src/run_scripts/startup/containerlaunch.sh"


iset() {
    [[ "$1" ]] && echo "true" || echo ""
}

truthy() {
    if [[ "$1" == "true" || "$1" == 1 ]]; then
        echo "true"
    elif [[ "$1" == "false" || "$1" == 0 ]]; then
        echo ""
    else
        iset $1
    fi
}


isint() {
    re='^[0-9]+$'
    if  [[ $1 =~ $re ]] ; then
       echo "true"
    fi
}

to () {
        cd $1
        ls --color=tty
}

f () {
        find . -iname '*'$1'*' 2> /dev/null
}

hist() { history | tail -$1
}

whois() {
    ping -c 1 $1
}

myip() {
    lsip | grep -v 127.0
}


dg() {
# docker grep for name
        docker ps | grep ${1} | awk '{ print $1 }'
}

dn() {
        if [ -z "$1" ]; then
                docker ps --format "{{.Names}}"
        else
                docker ps --format "{{.Names}}" | tail -n"+$1" | head -n1
        fi
}

dlc() {
    if [[ -z ${1} ]]; then
        NAME=$(dn 1) # grab the first container if no number specified
    elif [[ -n $(isint ${1}) ]] ; then
        NAME=$(dn ${1})
    else
        NAME=$(dg ${1})
    fi
    echo "${NAME}" | head -1
}

dosh() {
    NAME=$(dlc ${1})
    echo "starting ${NAME}"
    DOCKNUM=${1:-1}
    ENTRYPOINT=${2:-bash}
    docker exec -it ${NAME} ${ENTRYPOINT}
}

dofl() {
    docker logs -f $(dlc ${1})

}

gosh() {
        DOCKNAME=$( dg ${1} )
        ENTRYPOINT=${2:-bash}
        docker exec -it -e "DISPLAY" -e "QT_X11_NO_MITSHM=1" ${DOCKNAME} ${ENTRYPOINT}
}

doshg() {
# like dosh(), but passes in gui info
        DOCKNUM=${1:-1} # grab the first container if no number specified
        ENTRYPOINT=${2:-bash}
        docker exec -it -e "DISPLAY" -e "QT_X11_NO_MITSHM=1" `dn $DOCKNUM` ${ENTRYPOINT}
}


errcho() {
    (>&2 echo -e "\e[31m$1\e[0m")
}

ipgrep() {
    echo "$1" | grep -Eo '([0-9]*\.){3}[0-9]*'
}

ipfilt() {
  while read data; do
    ipgrep "$data"
  done
}

pingh() {
    res=$(ping -c1 "$1" 2>/dev/null)
    ips=$(echo "$res" | ipfilt | uniq)
    if [[ "$ips" ]]; then
        echo -e "\e[32m\e[1m$1:\e[0m $ips"
    else
       (>&2 echo -e "\e[31m host \e[4m$1\e[24m not found \e[0m")
    fi
}

triplicate() {
    # DONT WORK
    echo "$@"
    bash -c "$@"
    ssh nuvo1 bash -c "$@"
    ssh nuvo2 bash -c "$@"
}
