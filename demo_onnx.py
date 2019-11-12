from __future__ import print_function

import json
import os

import click
import onnxruntime
from PIL import Image

from util import data_util
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

    # load model
    yolo_session = onnxruntime.InferenceSession(model_config['onnx_file_path'])
    t.log_and_restart('load model')

    # prepare input
    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(model_config['input_resolution'])
    # Load an image from the specified input path, and return it together with  a pre-processed version
    image_raw, image = preprocessor.process(input_image)
    # Store the shape of the original input image in WH format, we will need it for later
    shape_orig_WH = image_raw.size
    t.log_and_restart('pre-process')

    # do inference
    input_name = yolo_session.get_inputs()[0].name
    onnx_outputs = yolo_session.run(None, input_feed={input_name: image})
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
    boxes, classes, scores = postprocessor.process(onnx_outputs,
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
