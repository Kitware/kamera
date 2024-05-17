#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
from pathlib import Path
import pydantic


def server_parser(description="general purpose server interface"):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        "--config_uri",
        default=None,
        action="store",
        type=str,
        help="input config file",
    )
    parser.add_argument(
        "-i",
        "--input_uri",
        default=None,
        action="store",
        type=str,
        help="an input file path",
    )
    parser.add_argument(
        "-H", "--host", default=None, action="store", type=str, help="host name"
    )
    parser.add_argument(
        "-p", "--port", default=8987, action="store", type=int, help="port"
    )
    parser.add_argument(
        "-+", "--health", action="store_true", help="Run a health check"
    )
    parser.add_argument(
        "-D", "--debug", action="store_true", help="Start in debug mode"
    )

    return parser


class ServerArgs(pydantic.BaseSettings):
    host: Optional[str]
    port: int = 8987
    config_uri: Optional[Path]
    input_uri: Optional[Path]
    health: bool = False
    debug: bool = False

    class Config:
        extra = "allow"
