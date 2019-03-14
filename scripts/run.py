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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import numpy as np
import cv2
import os
import tensorflow as tf
import shutil

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX 

capture = cv2.VideoCapture(0)
capture.set(3, 640) # set video widht
capture.set(4, 480) # set video height

try:
  os.mkdir('dump')
except:
  print("dump exists")

modelname = "tf_files/retrained_graph.pb"

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
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

while True:
  ret, img = capture.read()
  img = cv2.flip(img, +1)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces_color = faceCascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
  i = 0

  for(x,y,w,h) in faces_color:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,1), 1)    
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]   
    i = 0
    filename = "dump/" + str(i) + ".jpg"
    cv2.imwrite(filename, img[y:y+h,x:x+w])
    if __name__ == "__main__":
      file_name = filename
      model_file = modelname
      label_file = "tf_files/retrained_labels.txt"
      input_height = 224
      input_width = 224
      input_mean = 128
      input_std = 128
      input_layer = "input"
      output_layer = "final_result"
    
      parser = argparse.ArgumentParser()
      parser.add_argument("--image", help="image to be processed")
      parser.add_argument("--graph", help="graph/model to be executed")
      parser.add_argument("--labels", help="name of file containing labels")
      parser.add_argument("--input_height", type=int, help="input height")
      parser.add_argument("--input_width", type=int, help="input width")
      parser.add_argument("--input_mean", type=int, help="input mean")
      parser.add_argument("--input_std", type=int, help="input std")
      parser.add_argument("--input_layer", help="name of input layer")
      parser.add_argument("--output_layer", help="name of output layer")
      args = parser.parse_args()
    
      if args.graph:
        model_file = args.graph
      if args.image:
        file_name = args.image
      if args.labels:
        label_file = args.labels
      if args.input_height:
        input_height = args.input_height
      if args.input_width:
        input_width = args.input_width
      if args.input_mean:
        input_mean = args.input_mean
      if args.input_std:
        input_std = args.input_std
      if args.input_layer:
        input_layer = args.input_layer
      if args.output_layer:
        output_layer = args.output_layer
    
      graph = load_graph(model_file)
      t = read_tensor_from_image_file(file_name,
                                      input_height=input_height,
                                      input_width=input_width,
                                      input_mean=input_mean,
                                      input_std=input_std)
    
      input_name = "import/" + input_layer
      output_name = "import/" + output_layer
      input_operation = graph.get_operation_by_name(input_name);
      output_operation = graph.get_operation_by_name(output_name);
    
      with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                          {input_operation.outputs[0]: t})
        end=time.time()
      results = np.squeeze(results)
    
      top_k = results.argsort()[-5:][::-1]
      labels = load_labels(label_file)
    
      print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
      template = "{} (score={:0.5f})"
      for i in top_k:
        print(template.format(labels[i], results[i]))
      
      top = top_k[0]

      cv2.putText(img, str(labels[top]), (x+5,y-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
      cv2.putText(img, "{:0.2f} %".format(results[top]*100), (x+5,y+h-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1) 


  cv2.imshow("Face Recognition",img)
  keypress = cv2.waitKey(30)
  if keypress == 27: # press 'ESC' to quit
    break
  

capture.release()
shutil.rmtree('dump')       # To delete dump
cv2.destroyAllWindows()
  
