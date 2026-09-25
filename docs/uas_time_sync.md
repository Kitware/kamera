# How the uas systems kept time

The uas platform was removed from the tree in September 2026. Everything it had is still readable at the `uas-last` tag, for example:

```
git show uas-last:src/cfg/uas/uas0/ptp/chrony.conf
git show uas-last:tmux/uas/leader/03_run_clock_sync.sh
```

The part most worth remembering is how the three uas boxes and their cameras were kept on one clock. The crewed systems never did this. They run chrony against the INS host and take camera timestamps from the trigger event. The uas cameras were timestamped inside the camera, so every clock on the network had to agree.

## The chain, from GPS to camera

**1. GPS and PPS into the leader.** `gpsd` on uas0 read the GPS receiver on the serial port and the pulse per second signal on `/dev/pps3`. It published both into shared memory. Config: `src/cfg/uas/uas0/ptp/gpsd`.

**2. chrony set the leader's system clock.** chrony on uas0 used two shared memory reference clocks. The GPS sentence gave the absolute time and was marked coarse, with a 25 ms offset. The PPS gave the exact second edge and was marked good to 100 ns. chrony also served time to any host on the network that asked. Config: `src/cfg/uas/uas0/ptp/chrony.conf`.

**3. The leader's network card followed the system clock.** `phc2sys` pushed the system clock into the hardware clock of the network port facing the cameras. `ptp4l` ran on that port as the PTP grandmaster, with priority 0 so nothing could outrank it. Configs: `src/cfg/uas/uas0/ptp/phc2sys.service`, `ptp4l.service` and `ptp4l.conf`.

**4. The followers listened to the leader.** On uas1 and uas2, `ptp4l` ran listen only on the port facing the leader. `phc2sys` ran the other way round, copying the network card's clock into the system clock. The numbered start scripts under `tmux/uas` did the same job at supervisor level, so a follower coming up under supervisor synced itself the same way. Configs: `src/cfg/uas/uas1/ptp/` and `uas2/ptp/`, and `tmux/uas/follower/03_run_clock_sync.sh`.

**5. The cameras got PTP directly.** Each follower also ran a second `ptp4l` as a master on its RGB camera port, using `rgb.conf`. The Prosilica camera on that port took its time from the host, which by then matched the leader, which matched GPS. Script: `tmux/uas/follower/04_run_cam_ptp.sh`.

**6. Checking it.** `watch_time.sh` showed, every five seconds, chrony's sources, the PPS shared memory monitor, and the offset between the network card clock and the system clock. If all three looked right, the chain was intact. Script: `src/cfg/uas/watch_time.sh`.

## If it comes back

A future uas would be built on the current runtime layout under `tmux/`, not on the old one. The clock chain above is independent of that layout. It needs `gpsd`, `chrony` and `linuxptp` installed, a network card with a hardware clock on each host, and a GPS receiver with a PPS output on the leader. The old provisioning under `provision/ansible/playbooks/uas` at the tag shows how those were installed.
