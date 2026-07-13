# Changelog

All notable changes to KAMERA are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Dropdown selectors for Phase One shutter speed, gain, and aperture, with proper
  electronic/leaf shutter mapping.
- The 'Set' button lights up whenever a camera parameter changes, making pending
  changes obvious.
- Optional GPU JPEG compression via nvJPEG, enabled with the `/sys/arch/use_nvjpeg` flag.
- Power management (system startup/shutdown/reboot) as a simple stateful REST API,
  managed by kamerad and supervisor.
- kamerad runs on system startup and auto-restarts.
- Default camera configuration JSONs ship with the repo; defaults load based on the
  system name.
- Additional color status indicators in the GUI, and a warning when no detector
  pipefile is configured.
- Ansible tasks for system stop/start and docker restarts.
- Local docker registry on leader machines for distributing images to the rest of
  the system; leader machines also build the core and VIAME images.
- Tailscale installed with the system for remote access.
- Desktop shortcuts installed during provisioning, including a convenience symlink
  to the NAS.
- Script to seed the redis configuration during provisioning.
- nayak system support: tailored configuration (no IR/UV cameras) and a local
  stratum-3 NTP server so systems stay synchronized offline.

### Changed

- GUI ported to ROS Noetic: all images now build on Ubuntu 20.04 with Python 3 and
  wxPython 4.
- Save is now a single button: it finalizes changes, writes camera configurations to
  disk, and refreshes the main panel dropdown.
- Shutter speed is displayed directly instead of exposure in milliseconds.
- Flight number entry is more forgiving and no longer errors on every change.
- Contrast control is greyed out when the server manages it; disabled controls now
  show their values in a clearer disabled style.
- GUI layout overhauled: INS data in two columns, wider camera dropdowns, working
  About dialog, and many scaling fixes to eliminate clipped text and squished panels.
- Icons refreshed and properly labeled.
- `kamera_run` now just launches the GUI when the system is already up, so a
  separate GUI desktop icon is no longer needed.
- Hostnames follow a new center/left/right convention (hyphen-free for ROS
  compatibility), and start/stop scripts use it uniformly.
- Startup calls tmux directly, and environment/image-tag configuration is
  consolidated into supervisord globals and a `.env` file.
- Clear configuration boundaries: yaml files hold static, system-specific options;
  `system_state.json` holds mutable, GUI-facing state.
- Configuration conflict resolution reports what changed instead of failing silently.
- redis configuration consolidated.
- Docker build chain cleaned up: concrete makefile steps and explicit compose builds
  that respect `depends_on` order.
- NFS version pinned and NAS mount points created ahead of mounting, for more
  reliable NAS access.
- GUI relicensed under Apache 2.0; old license headers removed.

### Fixed

- INS panel incorrectly showing yellow (string vs. int comparison).
- ~10-frame lag in the image counter display; the frame buffer is also flushed when
  archiving stops, so counts stay accurate.
- GUI no longer hangs during SSD / NAS health checks (moved to a background thread).
- Race condition where the detector gauge started red.
- Intermittent X Windows errors when launching the GUI (containers now use
  `ipc: host`).
- Phase One debayering failures: added error handling around queue writes and
  removed a hardcoded output path.
- The image-processing queue is recreated on each run, guaranteeing every file gets
  processed.
- Detector pipefile loading is more robust.

### Removed

- Legacy ROS Kinetic / Python 2 code paths and shims.
- tmuxinator dependency.

## [0.4.0] - 2025-07-28

_Released before this changelog was introduced; see the git history for details._

## [0.3.0] - 2025-03-25

_Released before this changelog was introduced; see the git history for details._

## [0.2.0] - 2025-01-12

_Released before this changelog was introduced; see the git history for details._

## [0.1.0] - 2024-11-23

_Released before this changelog was introduced; see the git history for details._

[Unreleased]: https://github.com/Kitware/kamera/compare/v0.4.0...develop
[0.4.0]: https://github.com/Kitware/kamera/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Kitware/kamera/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Kitware/kamera/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Kitware/kamera/releases/tag/v0.1.0
