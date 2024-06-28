import cv2
import numpy as np
import os
import argparse
import time
import sys
from threading import Thread
import importlib.util
import multiprocessing
import tensorflow
import serial.tools.list_ports

ports = serial.tools.list_ports.comports()
serialInst = serial. Serial()
portsList =[]
portVar = "COM3"
serialInst.baudrate = 9600
serialInst.port = portVar
serialInst.open()
flag =0





# Define VideoStream class to handle streaming of video from webcam in separate processing thread
class VideoStream:
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(1)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.8)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,num_threads=multiprocessing.cpu_count(),
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,num_threads=multiprocessing.cpu_count())

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()

servo_x,servo_y=640,360

# def mapFunc(x,inmin,inmax,outmax,outmin):
    
#     return (((x-inmin)*(outmax-outmin)/(inmax-inmin))+outmin)

def map_value(fire_pixel, min_pixel, max_pixel, min_target=105, max_target=23):
    mapped_value = np.interp(fire_pixel, (min_pixel, max_pixel), (min_target, max_target))
    return mapped_value

def map_value_vertical(fire_pixel, min_pixel, max_pixel, min_target=50, max_target=102):
    mapped_value = np.interp(fire_pixel, (min_pixel, max_pixel), (min_target, max_target))
    return mapped_value
prev_x=0
prev_y=0
limit_box=[(500,0),(500,720),(900,0),(900,720)]
objects=[]
objects2=[{
    "coordinates":(0,0)
    }]
prev_dist=0
# Main loop for processing frames
while True:
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame = videostream.read()

    



    
   
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

   
   # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if flag == 0: 
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                center_x,center_y=(xmax+xmin)/2,(ymax+ymin)/2
                x,y=center_x-servo_x,center_y-servo_y
                duty_x=map_value(x,-640,640)
                duty_y=map_value_vertical(y,-360,360)
                duty_x=int(duty_x)
                duty_y=int(duty_y)

                print("=============================================")
                print("x value\t\t\t\ty vlaue")
                print(x,"\t\t\t\t",y)
                print("{:.2f}".format(duty_x),"\t""\t""\t","{:.2f}".format(duty_y))
                #print(flag)
                #to break loop after detection
                #flag=1
                #print(flag)    



            #------------------arduino communication-------------------   
                
                
                time.sleep(0.1)
                message= "X"+str(int(duty_x))+"Y"+str(int(duty_y))
               #message= "X1"+"Y"+str(int(duty_y))
                serialInst.write(message.encode('utf-8'))

                time.sleep(0.1)
                print("done--------------------")
                flag=1
                #serialInst.close()

                print(message)




                
                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i]*100))
                
                
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                
                
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
    
    if flag == 1:
        time.sleep(5)
        print(flag)
        # serialInst.open()
        # reply = serialInst.readline()
        # #time.sleep(0.1)
        # serialInst.close()
        # print(reply)
        flag = 0



    #print('----objects',objects)
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    #cv2.line(frame,limit_box[0],limit_box[1],(0,0,255),3)
    #cv2.line(frame,limit_box[2],limit_box[3],(0,0,255),3)
    # All the results have been drawn on the frame, so it's time to display it.






    #flipped_frame = cv2.flip(frame, 1)

    # Display the processed frame
    cv2.imshow('Object detector', frame)

    # Calculate and display framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1
    #print('FPS: {:.2f}'.format(frame_rate_calc))

    # Check for user input to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
