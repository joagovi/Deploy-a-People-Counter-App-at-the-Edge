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
#how to measure time in python taken from:
#https://stackoverflow.com/questions/7370801/how-to-measure-elapsed-time-in-python
from timeit import default_timer as timer

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

def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    conf_avg=0
    count_person=0;
    for box in result[0][0]: # Output shape is 1x1x100x7
        
        if (box[1]==1):
            conf = box[2]
            print(conf)
            if conf >= args.prob_threshold:
                count_person+=1
                conf_avg+=conf
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), convert_color('BLUE'), 1)
    return frame,count_person,conf_avg

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
    infer_network.load_model(model=args.model,cpu_extension=args.cpu_extension)

    ### TODO: Handle the input stream ###

    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)  
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    #out = cv2.VideoWriter('out2.mp4', 0x00000021, 30, (width,height)) Used for create an Ouput video file
    counter=0
    start_flag=0
    time_start=0
    count_person=0
    total_count_person=0
    last_count=0
    
    elapsed=0
    elapsed_prom=0
    frame_out=0
    time_counter=0
    conf_prom=0
    single_image_mode=0
    count_frame_person_total=0

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        counter+=1
        time_counter+=1

        ### TODO: Read from the video capture ###
        frame_prev_out=frame_out
        flag, frame = cap.read()

        if not flag:
            if (counter==2):
                single_image_mode=1
            break
            
        ### TODO: Pre-process the image as needed ###
        shape_input=infer_network.get_input_shape()       
        frame_proc=cv2.resize(frame,(shape_input[3],shape_input[2]))
        frame_proc=np.transpose(frame_proc,(2,0,1))
        frame_proc=np.reshape(frame_proc,(1,3,shape_input[2],shape_input[3]))

        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(frame_proc)
        
        ### It's use for measuring the inference time
        start = timer()
        ### TODO: Wait for the result ###
        if infer_network.wait()==0:
            end = timer()
            elapsed=(end - start)
            elapsed_prom=(elapsed_prom+elapsed)
            print(elapsed)

            ### TODO: Get the results of the inference request ###
            output_boxes=infer_network.get_output()
                
            ### TODO: Extract any desired stats from the results ###
            #This part has been adapted from: https://knowledge.udacity.com/questions/139281
            frame_out,count_person,conf=draw_boxes(frame,output_boxes,args,width,height)
            if(count_person>0):
                conf_prom+=conf
                count_frame_person_total+=count_person
            
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

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
            #out.write(frame) Used for create an Ouput video file

            ### TODO: Send the frame to the FFMPEG server ###
            sys.stdout.buffer.write(frame_out)
            sys.stdout.flush()

    ### TODO: Write an output image if `single_image_mode` ###
    if(single_image_mode==1):
        cv2.imwrite("/home/workspace/resources/out.png",frame_prev_out)
    
    #print(elapsed_prom/(time_counter-1))
    #print(conf_prom/count_frame_person_total)
    #out.release()
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()


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
