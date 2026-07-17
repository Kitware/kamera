#!/usr/bin/env python3
"""Seed Redis with system configuration.

Safe to re-run: existing keys are not overwritten unless --overwrite is passed.
Intended for use in Ansible provisioning. Requires only: redis-py, PyYAML.

    pip install redis pyyaml

Each --config file is loaded and its contents seeded under --prefix (default /sys).
Multiple --config files are merged in order; later files do not overwrite earlier ones
unless --overwrite is set.

Reimplements some logic in roskv to avoid building and importing.

Example:
    seed_redis.py --redis-host nuvo0 \\
        --config src/cfg/taiga/default_system_state.json
    seed_redis.py --redis-host nuvo0 \\
        --prefix /debug --config src/cfg/debug_defaults.json
"""
from __future__ import print_function
import argparse
import json
import os
from collections.abc import Mapping

import redis
import yaml


def _flatten(d, prefix="", sep="/"):
    """Recursively flatten a nested dict into {keypath: leaf_value} pairs."""
    items = {}
    for k, v in d.items():
        new_key = prefix + sep + k if prefix else k
        if isinstance(v, Mapping):
            items.update(_flatten(v, new_key, sep))
        else:
            items[new_key] = v
    return items


def seed(client, prefix, val, overwrite=False):
    """Write a nested dict into Redis under prefix, skipping existing keys."""
    if isinstance(val, Mapping):
        flat = _flatten(val, prefix=prefix)
    else:
        flat = {prefix: val}

    written = skipped = 0
    for k, v in flat.items():
        encoded = json.dumps(v) if not isinstance(v, str) else v
        if overwrite:
            client.set(k, encoded)
            written += 1
        else:
            if client.setnx(k, encoded):
                written += 1
            else:
                skipped += 1
    return written, skipped


def load_file(path):
    ext = os.path.splitext(path)[1].lower()
    with open(path) as f:
        if ext in (".yml", ".yaml"):
            return yaml.safe_load(f)
        elif ext == ".json":
            return json.load(f)
        else:
            raise ValueError("Unsupported file type: {}".format(path))


def main():
    parser = argparse.ArgumentParser(
        description="Seed Redis with system configuration (idempotent)."
    )
    parser.add_argument(
        "--redis-host",
        default=os.environ.get("REDIS_HOST", "localhost"),
        help="Redis hostname (default: $REDIS_HOST or localhost)",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=int(os.environ.get("REDIS_PORT", 6379)),
        help="Redis port (default: $REDIS_PORT or 6379)",
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        metavar="FILE",
        required=True,
        help="Config file to seed (.json or .yaml). May be specified multiple times.",
    )
    parser.add_argument(
        "--prefix",
        default="/sys",
        help="Redis key prefix to seed under (default: /sys)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Redis keys (use only for fresh provision)",
    )
    args = parser.parse_args()

    client = redis.Redis(
        host=args.redis_host, port=args.redis_port, client_name="provisioner"
    )

    total_written = total_skipped = 0
    for path in args.configs:
        data = load_file(path)
        written, skipped = seed(client, args.prefix, data, args.overwrite)
        print("  {}: {} written, {} skipped".format(path, written, skipped))
        total_written += written
        total_skipped += skipped

    print("Done. {} keys written, {} skipped (already set) on {}:{}".format(
        total_written, total_skipped, args.redis_host, args.redis_port
    ))


if __name__ == "__main__":
    main()
