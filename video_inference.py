from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

import numpy as np
import tensorflow as tf
import cv2

from picamera.array import PiRGBArray
from picamera import PiCamera

camx, camy, fps=224, 224, 30

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_img(img, input_height=299, input_width=299,
             input_mean=0, input_std=255):
  
  float_caster = tf.cast(img, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


if __name__ == "__main__":

  camera = PiCamera()
  camera.resolution = (camx,camy)
  camera.framerate = fps
  rawCapture = PiRGBArray(camera, size=(camx,camy))

  model_file = "retrained_graph.pb"
  label_file = "retrained_labels.txt"
##  input_height = 299
##  input_width = 299
##  input_mean = 0
##  input_std = 255
##  input_layer = "Mul"
  input_height = camy
  input_width = camx
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"

  graph = load_graph(model_file)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);

  labels = load_labels(label_file)

  time_array=np.zeros(30,dtype=np.float)
  
  for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    
    frame = image= frame.array
    rawCapture.truncate(0)

    t = read_img(frame,
                 input_height=input_height,
                 input_width=input_width,
                 input_mean=input_mean,
                 input_std=input_std)
    
    with tf.Session(graph=graph) as sess:
      start = time.time()
      results = sess.run(output_operation.outputs[0],
                        {input_operation.outputs[0]: t})
      end=time.time()
    results = np.squeeze(results) 

    top_k = results.argsort()[-5:][::-1]
    
    print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
    template = "{} (score={:0.5f})"
    for i in top_k:
      print(template.format(labels[i], results[i]))

    cv2.imshow('frame',frame.astype(np.uint8))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  cv2.destroyAllWindows()
