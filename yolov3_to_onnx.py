#!/usr/bin/env python2

from __future__ import print_function

import json
import sys
from collections import OrderedDict

import click
import onnx

from util.onnx_util import DarkNetParser, GraphBuilderONNX


@click.command()
@click.argument('config_path', required=True, type=click.Path(exists=True))
def main(config_path):
    # Have to use python 2 due to hashlib compatibility
    if sys.version_info[0] > 2:
        raise Exception(
            'This script is only compatible with python2, '
            'please re-run this script with python2. '
            'The rest of this sample can be run with either version of python.'
        )

    # Load config
    with open(config_path) as f:
        model_config = json.load(f)

    # These are the only layers DarkNetParser will extract parameters from. The three layers of
    # type 'yolo' are not parsed in detail because they are included in the post-processing later:
    supported_layers = [
        'net', 'convolutional', 'shortcut', 'route', 'upsample', 'maxpool'
    ]

    # Create a DarkNetParser object, and the use it to generate an OrderedDict with all
    # layer's configs from the cfg file:
    parser = DarkNetParser(supported_layers)
    layer_configs = parser.parse_cfg_file(model_config['cfg_file_path'])
    # We do not need the parser anymore after we got layer_configs:
    del parser

    # In above layer_config, there are three outputs that we need to know the output
    # shape of (in CHW format):
    output_tensor_dims = OrderedDict(
        sorted(model_config['output_tensor_dims'].items()))

    # Create a GraphBuilderONNX object with the known output tensor dimensions:
    builder = GraphBuilderONNX(output_tensor_dims)

    # Now generate an ONNX graph with weights from the previously parsed layer configurations
    # and the weights file:
    yolov3_model_def = builder.build_onnx_graph(
        layer_configs=layer_configs,
        weights_file_path=model_config['weights_file_path'],
        verbose=True)
    # Once we have the model definition, we do not need the builder anymore:
    del builder

    # Perform a sanity check on the ONNX model definition:
    onnx.checker.check_model(yolov3_model_def)

    # Serialize the generated ONNX graph to this file:
    onnx.save(yolov3_model_def, model_config['onnx_file_path'])


if __name__ == '__main__':
    main()
