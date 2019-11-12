#!/usr/bin/env python3

from __future__ import print_function

import json

import click

from util.trt_util import get_engine


@click.command()
@click.argument('config_path', required=True, type=click.Path(exists=True))
def main(config_path):
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    # Load config
    with open(config_path) as f:
        model_config = json.load(f)

    # Build engine
    get_engine(model_config['onnx_file_path'],
               model_config['trt_file_path'],
               model_config['trt_fp16_mode'],
               overwrite=True)


if __name__ == '__main__':
    main()
