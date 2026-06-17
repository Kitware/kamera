#! /usr/bin/python
"""Seed Redis with the static system config from config.yaml.

kamcore nodes (cam_param_monitor, fps_monitor, ...) read /sys/arch/* and
/sys/channels straight from Redis, but nothing in the core startup populates
them -- historically that happened only as a side effect of the GUI importing
wxpython_gui.cfg, so the nodes raced (and crashed against) the GUI on a fresh
boot. This seeds the static config up front so they no longer depend on it.

Only config.yaml (the static deployment truth) is written; operator-mutable
session state (flight, project, effort, ...) stays owned by the GUI. config.yaml
keys are authoritative, so overwriting any stale Redis values here is correct.
"""
import os
import sys

import yaml

from roskv.impl.redis_envoy import RedisEnvoy


def main():
    if len(sys.argv) > 1:
        cfg_file = sys.argv[1]
    else:
        cfg_file = "/cfg/%s/config.yaml" % os.environ["SYSTEM_NAME"]

    with open(cfg_file, "r") as stream:
        config = yaml.safe_load(stream)

    envoy = RedisEnvoy(os.environ["REDIS_HOST"], client_name="config_seeder")
    # config.yaml is authoritative for its static keys. Clear each static subtree
    # first so keys removed from the yaml (e.g. a dropped channel) don't linger
    # in Redis -- put only sets keys, it never deletes. "arch" also carries the
    # GUI's mutable session state (flight, project, ...), so only its static
    # "hosts" subtree is reset there, never all of /sys/arch.
    for key, val in config.items():
        if isinstance(val, dict) and key != "arch":
            try:
                envoy.delete_dict("/sys/%s" % key)
            except Exception:
                pass
    try:
        envoy.delete_dict("/sys/arch/hosts")
    except Exception:
        pass
    # Mirror wxpython_gui.cfg: each top-level config.yaml key is published under
    # /sys (RedisEnvoy.put flattens nested dicts into keypaths).
    for key, val in config.items():
        envoy.put("/sys/%s" % key, val)
    print("Seeded /sys from %s (%d top-level keys)" % (cfg_file, len(config)))


if __name__ == "__main__":
    main()
