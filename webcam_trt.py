#!/usr/bin/env python3

from __future__ import print_function

import json

import click
import cv2
import numpy as np

from util import data_util, trt_util
from util.data_util import PreprocessYOLO, PostprocessYOLO
from util.timer import Timer


def gstreamer_pipeline(
        capture_width=1280,
        capture_height=720,
        display_width=1280,
        display_height=720,
        framerate=30,
        flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink" % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        ))


@click.command()
@click.argument('config_path', required=True, type=click.Path(exists=True))
def main(config_path):
    # load config
    with open(config_path) as f:
        model_config = json.load(f)

    # initialize timer
    t = Timer()

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0),
                           cv2.CAP_GSTREAMER)

    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Create a pre-processor object by specifying the required input resolution for YOLOv3
        preprocessor = PreprocessYOLO(model_config['input_resolution'])
        # Create a post-processor object
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
        # Output shapes expected by the post-processor
        output_shapes = model_config['output_shapes']
        # initialize enfine
        trt_outputs = []
        # Window
        with trt_util.get_engine(
                model_config['onnx_file_path'],
                model_config['trt_file_path'],
                model_config['trt_fp16_mode'],
        ) as engine, engine.create_execution_context() as context:
            t.log_and_restart('load model')
            inputs, outputs, bindings, stream = trt_util.allocate_buffers(
                engine)
            while cv2.getWindowProperty("CSI Camera", 0) >= 0:
                ret_val, input_image = cap.read()
                if input_image is None:
                    break

                # Load an image from the specified input path, and return it together with  a pre-processed version
                image_raw, image = preprocessor.process(input_image)
                # Store the shape of the original input image in WH format, we will need it for later
                shape_orig_WH = image_raw.size
                t.log_and_restart('pre-process')

                # Do inference
                # print('Running inference on image...'.format(input_image))
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

                # Run the post-processing algorithms on the TensorRT outputs
                # and get the bounding box details of detected objects
                boxes, classes, scores = postprocessor.process(
                    trt_outputs, (shape_orig_WH))
                t.log_and_restart('post-process')

                # Let's make sure that there are 80 classes, as expected for the COCO data set:
                all_categories = data_util.load_label_categories(
                    model_config['label_file_path'])
                assert len(all_categories) == model_config['num_classes']

                # Draw the bounding boxes onto the original input image and save it as a PNG file
                obj_detected_img = data_util.draw_bboxes(
                    image_raw, boxes, scores, classes, all_categories)
                t.log_and_restart('visualize')

                img_to_display = np.array(obj_detected_img)
                cv2.imshow("CSI Camera", img_to_display)
                # This also acts as
                keyCode = cv2.waitKey(30) & 0xFF
                # Stop the program on the ESC key
                if keyCode == 27:
                    break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")

    # Do inference with TensorRT

    t.print_log()


if __name__ == '__main__':
    main()
