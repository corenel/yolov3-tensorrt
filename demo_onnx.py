from __future__ import print_function

import click
import numpy as np
import onnxruntime
from PIL import Image, ImageDraw

from util.data_util import load_label_categories


@click.command()
@click.argument('config_path', required=True, type=click.Path(exists=True))
@click.option('-i',
              '--input-image',
              required=True,
              type=click.Path(exists=True))
def main(config_path, input_image):

    image = Image.open(input_image)
    image_data_onnx = prepare_data(
        image, model_image_size=model_config['input_resolution'])
    feed_f = dict(
        zip(['input_1', 'image_shape'],
            (image_data_onnx,
             np.array([image.size[1], image.size[0]], dtype='float32').reshape(
                 1, 2))))
    yolo_session = onnxruntime.InferenceSession(model_config['onnx_file_path'])
    all_boxes, all_scores, indices = yolo_session.run(None, input_feed=feed_f)

    out_boxes, out_scores, out_classes = [], [], []
    for idx_ in indices[0]:
        out_classes.append(idx_[1])
        out_scores.append(all_scores[tuple(idx_)])
        idx_1 = (idx_[0], idx_[2])
        out_boxes.append(all_boxes[idx_1])

    thickness = (image.size[0] + image.size[1]) // 300
    all_categories = load_label_categories(model_config['label_file_path'])
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = all_categories[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i],
                           outline=self.colors[c])
        draw.rectangle([tuple(text_origin),
                        tuple(text_origin + label_size)],
                       fill=self.colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw


if __name__ == '__main__':
    main()
