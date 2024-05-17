#!/bin/bash

# ansi color codes
BLU='\033[0;34m'
GRN='\033[0;32m'
RED='\033[0;31m'
PUR='\033[0;35m'
NC='\033[0m' # No Color

alias errcho='>&2 echo' # fancy
function errprint() {
	printf "$@" 1>&2
}

function dbgprint() {
	if false ; then
		printf "$@" 1>&2
	fi
}

# get the possible valid ip addresses for LAN connections and their interface names
unset -f my_ips
function my_ips () {
	lans='(192|10)\.'
	all_ips=$(ip -o addr | awk 'BEGIN{RS="\n"}{split($4, a, "/"); print $2" "a[1]}')
	printf "%s" "$all_ips" | grep -E "$lans" ; 
}

function get_master_interface () {
	num_ips=$(echo "$(my_ips)" | wc -l) ;
	dbgprint "${GRN} Found ${num_ips} IPs:\n${PUR}$(my_ips) ${NC}\n" ;
	if [ "$num_ips" -eq "0" ]; then
		errprint "${RED} NO IPS FOUND ${NC}\n" ;
		return 1 ;
	fi
	
	if [ "$num_ips" -gt "1" ]; then
		errprint "${RED} TOO MANY IPS FOUND ${NC}\n" ;
		return 2 ;
	fi
	dbgprint "${BLU} ${num_ips} IP FOUND ${NC}\n" ;
	printf "%s" "$(my_ips)" | awk '{print $1}'
}

function uber_arp () {
	if [ -z "$(which arp-scan)" ]; then
		apt update;
		apt install arp-scan;
	fi
	iface="$(get_master_interface)"
	if [ -z "$iface" ]; then
		errprint "Unable to find an appropriate interface. check stderr"
		return 1 ;
	fi 
	arp-scan --interface="$iface" --localnet
}

# script to find cameras' IP. Just grab the first one 
function get_flir_ip () {
	devname="Pleora"
	uber_arp | grep "$devname" | awk '{print $1}' | head -1
}
