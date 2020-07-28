#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = IECore()
        self.network = None
        self.exec_network = None
        self.input_blob = None
        self.output_blob = None
        self.input_info = None


    def load_model(self, model,cpu_extension):
        model_xml = model        
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        if cpu_extension:
            self.plugin.add_extension(cpu_extension, "CPU")
        
        ### TODO: Load the model ###
        self.network = IENetwork(model=model_xml, weights=model_bin)
        self.network.add_outputs(['detection_output'])
        self.exec_network = self.plugin.load_network(self.network, "CPU")
        
        ### TODO: Check for supported layers ###
        supported_layers = self.plugin.query_network(network=self.network, device_name="CPU")
        #print(supported_layers)        
        
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)
        
        ### TODO: Add any necessary extensions ###
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        #I understood the input form of the model thanks to:
        #https://knowledge.udacity.com/questions/196315
        #https://knowledge.udacity.com/questions/157050
        #https://knowledge.udacity.com/questions/137114
        inputs = iter(self.network.inputs)
        self.input_info = "image_info"
        self.input_blob= "image_tensor"
        self.output_blob = next(iter(self.network.outputs))
        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        #print(self.input_blob)
        #print(self.network.inputs[self.input_info].shape)
        #print(self.network.inputs)
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, image):
        ### TODO: Start an asynchronous request ###
        #print(self.network.inputs[self.input_info])
        #print(image.shape[1:])
        self.exec_network.requests[0].async_infer({self.input_info:[800,800,1],self.input_blob: image})
        #self.exec_network.start_async(request_id=0, inputs={self.input_blob: image})
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        status = self.exec_network.requests[0].wait(-1)
        return status

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[0].outputs['detection_output']#[self.output_blob]
