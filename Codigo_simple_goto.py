from _future_ import absolute_import
from _future_ import division
from _future_ import print_function

from dronekit import connect, VehicleMode
import time
import cv2
import argparse
import numpy as np
import tensorflow as tf


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  model_file = "/home/eliel/Documentos/ProyectoTitulacion/CodigosPython/ProyectoDrone/output_graph.pb"
  label_file = "/home/eliel/Documentos/ProyectoTitulacion/CodigosPython/ProyectoDrone/output_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "Placeholder"
  output_layer = "final_result"

  graph = load_graph(model_file)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)
  
  with tf.Session(graph=graph) as sess:
    results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: result
    })
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(label_file)
  
  for i in top_k:
	  print(labels[i], results[i])
  # print(top_k)
  return top_k[0]


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

# CAMARA
camera_port = 0
ramp_frames = 30
camera = cv2.VideoCapture(camera_port)
 
 def get_image():
  retval, im = camera.read()
  return im

for i in range(ramp_frames):
	temp = get_image()
print("Taking image/tomando imagen...")

camera_capture = get_image()
file="imagen.jpg"

cv2.imwrite(file, camera_capture)

del(camera)
print ('CONNECTING...')
api = local_connect()
vehicle = api.get_vehicles()[0]

labelPrint = read_tensor_from_image_file(file)
print(labelPrint)

if labelPrint == 0:
	print("Reconocio Cuadrado")
  print("Now let's land")
  vehicle.mode = VehicleMode("LAND")
  vehicle.close() 
  vehicle.armed = False
 
elif labelPrint == 1:
    print("Reconocio Triangulo")
    print("Armando Vehiculo")
vehicle.armed   = True
print ("vehiculo se armo correctamente")
    def arm_and_takeoff(aTargetAltitude):
                
            vehicle.armed   = True
            
            while not vehicle.armed:
                print (" Waiting for arming...")
                time.sleep(1)

            print ("Taking off!")
            vehicle.simple_takeoff(aTargetAltitude) 
            while True:
                print (" Altitude: "), vehicle.location.global_relative_frame.alt 
                if vehicle.location.global_relative_frame.alt>=aTargetAltitude*0.95: 
                    print ("Reached target altitude")
                    break
                time.sleep(1)
                
                arm_and_takeoff(10)
                print("Take off complete")
                time.sleep(10)
                