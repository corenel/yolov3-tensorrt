#!/usr/bin/env python3
#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

from __future__ import print_function
from collections import OrderedDict

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw

from yolov3_to_onnx import download_file
from data_processing import PreprocessYOLO, PostprocessYOLO, load_label_categories

import sys
import os
import json
import common
import click

TRT_LOGGER = trt.Logger()


def draw_bboxes(image_raw,
                bboxes,
                confidences,
                categories,
                all_categories,
                bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    draw = ImageDraw.Draw(image_raw)
    print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width,
                    np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height,
                     np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12),
                  '{0} {1:.2f}'.format(all_categories[category], score),
                  fill=bbox_color)

    return image_raw


def get_engine(onnx_file_path,
               engine_file_path="",
               fp16_mode=False,
               overwrite=False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28  # 256MB
            builder.max_batch_size = 1
            if fp16_mode:
                print('Using FP16 mode')
                builder.fp16_mode = True
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    'ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'
                    .format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.
                  format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            print(
                'Saving TensorRT file to path {}...'.format(engine_file_path))
            return engine

    if os.path.exists(engine_file_path) and not overwrite:
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path,
                  "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


@click.command()
@click.argument('config_path', required=True, type=click.Path(exists=True))
@click.option('-i', '--input-image', type=click.Path(exists=True))
def main(config_path, input_image):
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    # Load config
    with open(config_path) as f:
        model_config = json.load(f)

    if input_image is None:
        get_engine(model_config['onnx_file_path'],
                   model_config['trt_file_path'],
                   model_config['trt_fp16_mode'],
                   overwrite=True)
        return

    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(model_config['input_resolution'])
    # Load an image from the specified input path, and return it together with  a pre-processed version
    image_raw, image = preprocessor.process(input_image)
    # Store the shape of the original input image in WH format, we will need it for later
    shape_orig_WH = image_raw.size

    # Output shapes expected by the post-processor
    output_shapes = model_config['output_shapes']
    # Do inference with TensorRT
    trt_outputs = []
    with get_engine(
            model_config['onnx_file_path'],
            model_config['trt_file_path'],
            model_config['trt_fp16_mode'],
    ) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        print('Running inference on image {}...'.format(input_image))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = image
        trt_outputs = common.do_inference(context,
                                          bindings=bindings,
                                          inputs=inputs,
                                          outputs=outputs,
                                          stream=stream)

    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    trt_outputs = [
        output.reshape(shape)
        for output, shape in zip(trt_outputs, output_shapes)
    ]

    postprocessor_args = {
        # A list of 3 three-dimensional tuples for the YOLO masks
        "yolo_masks": model_config['masks'],
        "yolo_anchors": model_config['anchors'],
        # Threshold for object coverage, float value between 0 and 1
        "obj_threshold": model_config['obj_threshold'],
        # Threshold for non-max suppression algorithm, float value between 0 and 1
        "nms_threshold": model_config['nms_threshold'],
        "yolo_input_resolution": model_config['input_resolution'],
        "yolo_num_classes": model_config['num_classes']
    }

    postprocessor = PostprocessYOLO(**postprocessor_args)

    # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
    boxes, classes, scores = postprocessor.process(trt_outputs,
                                                   (shape_orig_WH))
    # Let's make sure that there are 80 classes, as expected for the COCO data set:
    all_categories = load_label_categories(model_config['label_file_path'])
    assert len(all_categories) == model_config['num_classes']

    # Draw the bounding boxes onto the original input image and save it as a PNG file
    obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes,
                                   all_categories)
    output_image_path = os.path.splitext(input_image)[0] + '_result.png'
    obj_detected_img.save(output_image_path, 'PNG')
    print('Saved image with bounding boxes of detected objects to {}.'.format(
        output_image_path))


if __name__ == '__main__':
    main()
