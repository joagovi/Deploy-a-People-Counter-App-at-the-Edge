"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

import numpy as np

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.3,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT,MQTT_KEEPALIVE_INTERVAL)

    return client

def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2, 0, 1))
    image = image.reshape(1, 3, height, width)

    return image

def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    count_person=0;
    #print(np.asarray(result[0][0]).shape)
    for box in result[0][0]: # Output shape is 1x1x100x7
        
        if (box[1]==1):
            count_person+=1
            conf = box[2]
            if conf >= args.prob_threshold:
                xmin = int(box[3] * width)
                #print(box[3])
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), convert_color('BLUE'), 1)
    return frame,count_person

def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Blue if an invalid color is given.
    '''
    colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors['BLUE']

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    #print(args.cpu_extension)
    #print(args.model)
    infer_network.load_model(model=args.model,cpu_extension=args.cpu_extension)

    ### TODO: Handle the input stream ###

    ### TODO: Loop until stream is over ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)  
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    out = cv2.VideoWriter('out2.mp4', 0x00000021, 30, (width,height))
    counter=0
    start_flag=0
    time_start=0
    count_person=0
    total_count_person=0
    last_count=0
    
    while cap.isOpened():
        counter+=1

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()

        if not flag:
            break
        #flag=0
        ### TODO: Pre-process the image as needed ###
        shape_input=infer_network.get_input_shape()
        #print(shape_input)
        #print(frame.shape)
       
        frame_proc=cv2.resize(frame,(shape_input[3],shape_input[2]))
        frame_proc=np.transpose(frame_proc,(2,0,1))
        frame_proc=np.reshape(frame_proc,(1,3,shape_input[2],shape_input[3]))
        #print(frame_proc.shape)

        ### TODO: Start asynchronous inference for specified request ###
        
        if(not((counter-1)%10)):
            infer_network.exec_net(frame_proc)
            
            ### TODO: Wait for the result ###
            infer_network.wait()
            
            ### TODO: Get the results of the inference request ###
            output_boxes=infer_network.get_output()
                
        ### TODO: Extract any desired stats from the results ###
        #output_boxes=output_boxes['detection_output']

        ### TODO: Calculate and send relevant information on ###
        #This part has been adapted from: https://knowledge.udacity.com/questions/139281
        ### current_count, total_count and duration to the MQTT server ###
        ### Topic "person": keys of "count" and "total" ###
        ### Topic "person/duration": key of "duration" ###

        frame_out,count_person=draw_boxes(frame,output_boxes,args,width,height)
        client.publish("person", json.dumps({"count": count_person}))
        
        if count_person > last_count:
            time_start=counter/10
            total_count_person = total_count_person + count_person - last_count
            client.publish("person", json.dumps({"total": total_count_person}))
        # Person duration in the video is calculated
        if  count_person < last_count:
            duration = int(counter/10 - time_start)
            counter=couter=0
            # Publish messages to the MQTT server
            client.publish("person/duration",json.dumps({"duration": duration}))
        
        last_count = count_person                       
        #out.write(frame)
        #print(time_start)
        #print(count_person)

    ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame_out)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()#quitar?


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
