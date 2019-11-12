#!/usr/bin/env python3

from __future__ import print_function

import json
import os

import click

from util import data_util, trt_util
from util.data_util import PreprocessYOLO, PostprocessYOLO
from util.timer import Timer


@click.command()
@click.argument('config_path', required=True, type=click.Path(exists=True))
@click.option('-i',
              '--input-image',
              required=True,
              type=click.Path(exists=True))
def main(config_path, input_image):
    # Load config
    with open(config_path) as f:
        model_config = json.load(f)

    # initialize timer
    t = Timer()

    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(model_config['input_resolution'])
    # Load an image from the specified input path, and return it together with  a pre-processed version
    image_raw, image = preprocessor.process(input_image)
    # Store the shape of the original input image in WH format, we will need it for later
    shape_orig_WH = image_raw.size
    t.log_and_restart('pre-process')

    # Output shapes expected by the post-processor
    output_shapes = model_config['output_shapes']
    # Do inference with TensorRT
    trt_outputs = []
    with trt_util.get_engine(
            model_config['onnx_file_path'],
            model_config['trt_file_path'],
            model_config['trt_fp16_mode'],
    ) as engine, engine.create_execution_context() as context:
        t.log_and_restart('load model')
        inputs, outputs, bindings, stream = trt_util.allocate_buffers(engine)
        # Do inference
        print('Running inference on image {}...'.format(input_image))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = image
        trt_outputs = trt_util.do_inference(context,
                                            bindings=bindings,
                                            inputs=inputs,
                                            outputs=outputs,
                                            stream=stream)

    # Before doing post-processing, we need to reshape the outputs
    # as the common.do_inference will give us flat arrays.
    trt_outputs = [
        output.reshape(shape)
        for output, shape in zip(trt_outputs, output_shapes)
    ]
    t.log_and_restart('inference')

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
    t.log_and_restart('post-process')

    # Let's make sure that there are 80 classes, as expected for the COCO data set:
    all_categories = data_util.load_label_categories(
        model_config['label_file_path'])
    assert len(all_categories) == model_config['num_classes']

    # Draw the bounding boxes onto the original input image and save it as a PNG file
    obj_detected_img = data_util.draw_bboxes(image_raw, boxes, scores, classes,
                                             all_categories)
    output_image_path = os.path.splitext(input_image)[0] + '_result.png'
    obj_detected_img.save(output_image_path, 'PNG')
    print('Saved image with bounding boxes of detected objects to {}.'.format(
        output_image_path))
    t.log_and_restart('visualize')

    t.print_log()


if __name__ == '__main__':
    main()
