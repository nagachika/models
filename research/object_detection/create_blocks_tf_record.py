# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert BLOCKS dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=gs://myBucket/images/ \
        --output_path=gs://myBucket/data/tf.record \
        --label_map_path=gs://myBucket/label_map.pbtxt
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import os.path as path
import json
import threading
import queue

import PIL.Image
import tensorflow as tf
import argparse

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


def dict_to_tf_example(data, label_map_dict):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding fields for a single image.
    label_map_dict: A map from string label names to integers ids.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  full_path = data['filename']
  with tf.io.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  image = image.resize((300, 300), PIL.Image.BICUBIC)
  buf = io.BytesIO()
  image.save(buf, "JPEG")
  encoded_jpg = buf.getvalue()
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  is_crowd_obj = []
  for obj in data['objects']:
    xmin.append(float(obj['bndbox']['xmin']))
    ymin.append(float(obj['bndbox']['ymin']))
    xmax.append(float(obj['bndbox']['xmax']))
    ymax.append(float(obj['bndbox']['ymax']))
    classes_text.append(obj['label'].encode('utf8'))
    classes.append(label_map_dict[obj['label']])
    truncated.append(0)
    is_crowd_obj.append(False)

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/is_crowd': dataset_util.int64_list_feature(is_crowd_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
  }))
  return example

def worker(q, output_path, label_map_dict, exceptions):
    task_popped = False
    logging.info("open writer to {}".format(output_path))
    writer = tf.io.TFRecordWriter(output_path)
    while True:
        try:
            idx, image_path = q.get()
            task_popped = True
            if image_path == None:
              logging.info("Flush writer to {}".format(output_path))
              writer.close()
              q.task_done()
              return
            with tf.io.gfile.GFile(image_path, 'r') as fid:
                json_str = fid.read()
            data = json.loads(json_str)
            tf_example = dict_to_tf_example(data, label_map_dict)
            if tf_example:
                writer.write(tf_example.SerializeToString())
            q.task_done()
            task_popped = False
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(examples_list))
        except Exception as e:
            logging.error("preprocessing image failed: {}: {}".format(type(e).__name__, e))
            if task_popped:
                q.task_done()
            exceptions.append(e)

def main():
  parser = argparse.ArgumentParser(description='')
  parser.add_argument("--data_dir", required=True, help="Root directory to JPEG images dataset.")
  parser.add_argument("--output_path", required=True, help="Path to output TFRecord")
  parser.add_argument("--label_map_path", required=True, help="Path to label map proto")
  FLAGS = parser.parse_args()

  data_dir = FLAGS.data_dir

  tf.io.gfile.makedirs(path.dirname(FLAGS.output_path))

  q = queue.Queue()
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  logging.info('Reading from dataset (JSON) in %s.', FLAGS.data_dir)
  examples_list = tf.io.gfile.glob(path.join(FLAGS.data_dir, "*.json"))

  if len(examples_list) == 0:
      raise ValueError("No annotation example was found.")

  for idx, example in enumerate(examples_list):
    q.put((idx, example))

  threads_num = min(10, len(examples_list))
  threads = []
  exceptions = []
  for i in range(threads_num):
    output_path = "%s_%05d" % (FLAGS.output_path, i+1)
    th = threading.Thread(target=worker, daemon=True, args=[q, output_path, label_map_dict, exceptions])
    threads.append(th)
    th.start()

  for _ in range(threads_num):
      q.put(None)

  logging.info('Wait all conversion done.')
  q.join()
  logging.info('All conversion done.')

  if len(exceptions) != 0:
      raise ValueError("Some image preprocessing failed.")

if __name__ == '__main__':
  main()
