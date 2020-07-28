# Project Write-Up

This document explains the first project of the  Intel® Edge AI for IoT developers Nanodegree: Deploy a People counter app on the edge. For development I have used the Udacity workspace.


The file to run the application using Openvino is [main.py](). Additionally, because the inference time in the workspace is long, I have considered a modification to this file. This aditional file is in the proyect [Proyect1_B]() and it makes one inference every 10 frames (1 second) and the frame is only sent to the ffmpeg server every second (The video is 10FPS).

## Explaining Custom Layers

***The process behind converting custom layers involves the following steps:***


First, keep in mind that the model must be a frozen model. For this case I have used the Mask RCNN model:

```console
wget http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet101_atrous_coco_2018_01_28.tar.gz
tar xvf mask_rcnn_resnet101_atrous_coco_2018_01_28.tar.gz

```

To do the conversion I have used the following:

```console
/opt/intel/openvino/deployment_tools/model_optimizer/mo.py --data_type=FP16 --reverse_input_channels --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/mask_rcnn_support.json

```

The config files [pipeline.config](https://github.com/Logeswaran123/Garbage-Classification-for-safety/tree/master/model_O_R) and [frozen_inference_graph.pb](https://github.com/Logeswaran123/Garbage-Classification-for-safety/tree/master/model_O_R) are inside the downloaded folder.

The inversion of the channels is used since the images of COCO (dataset with which the network was trained) are in RGB.

The subgraph configuration file [mask_rcnn_support.json](https://github.com/Logeswaran123/Garbage-Classification-for-safety/tree/master/model_O_R) was used successfully, the other provided files were tried but none gave a successful result.

Initially, the FP32 precision was used; however, the inference was very slow in the workspace and I decided to reduce it to 16 bits. Even so, I continue to be slow due to the limitations of the workspace.

After conversion to the intermediate representation, the supported layers were tested. Because there may be models whose capabilities are not supported, it is necessary to verify it.

Before adding the existing CPU extensions, I found unsupported layers. Subsequently, adding the following extension will make all layers supported for the model used.
```
/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so
```

If until this step I had found unsupported layers I would have had to use the layers of the same tensorflow or in any case use another subgraph that is supported and equivalent to the unsupported layer.

***Reasons for handling custom layers:***

There are layers of given frameworks that are not supported. In addition there are layers that can be supported only by some devices and not supported by the devices (Hardware) in which you are working. In this case, you can create custom layers, add extensions (hardware dependent), or work the layer in the original framework..

## Comparing Model Performance

***My method for comparing models before and after conversion to Intermediate Representations
involve the following.***

**For the model optimized using Openvino.**

For the evaluation, a portion of the video [Pedestrian_Detect_2_1_1.mp4]() has been taken into account for a 46 frame video size (about 4 seconds).

To run the model with Openvino and see the indicators, it is necessary uncomment lines 227 and 228 and comment lines 220 and 221 , to have the Mosca server active and execute the following command:

```console
(venv) root@907e6ba7b117:/home/workspace/mask-rcnn/mask_rcnn_resnet101_atrous_coco_2018_01_28# python /home/workspac
e/main.py -m frozen_inference_graph.xml  -i /home/workspace/resources/prueba1.mp4 -l /opt/intel/openvino/deployment_
tools/inference_engine/lib/intel64/libcpu_extension_sse4.so
```

**For the TensorFlow original model.**

The [TensorFlow Object Detection API example](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb) has been used.

This example has been modified to work with videos to get the comparison indicators, the modified notebook is [Objetc_detection_TF.ipynb]().

To install and use the object detection API in the Udacity workspace:

```console
pip install -U --pre tensorflow=="2.*"
pip install tf_slim
pip install pycocotools
```

Once the files were downloaded, before installing the API the [ops.py]() file had to be modified:

The line: 
```python
resize_method = 'nearest' if box_masks.dtype == tf.uint8 else resize_method
```
produces the following error:
```console
Cannot interpret 'tf.uint8' as a data type
```

To avoid the error, the line has been commented momentarily, since the precision is float32 this line is unnecessary in this case.


***Difference between model accuracy pre- and post-conversion***

For the video considered, the average accuracy pre- and post-conversion was 0.9795 y 0.9731 respectively. 
There is no significant reduction in precision.


***The size of the model***

The size of the model is cut in half. The size of the model before and after the conversion was 212M and 104M respectively.

```console
root@f9827b69fcc8:/home/workspace/mask-rcnn/mask_rcnn_resnet101_atrous_coco_2018_01_28# ls -lh
-rw-r--r-- 1 root   root 104M Jul 26 10:24 frozen_inference_graph.bin
-rw-r--r-- 1 root   root  76K Jul 26 10:24 frozen_inference_graph.mapping
-rw-r--r-- 1 345018 5000 212M Feb  1  2018 frozen_inference_graph.pb
```

***Difference between the inference time of the model pre- and post-conversion***

I've used [prueba1.mp4]() video to compare the average inference times. 


To run the main.py file in the workspace:
```console
python /home/workspace/main.py -m frozen_inference_graph.xml  -i /home/workspace/resources/prueba1.mp4 -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so
```

The average inference time is 56.56 seconds using TensorFlow and 32.2 seconds using openvino's optimized model with float32 quantization.

Inference time is significantly reduced. However, other factors must be taken into account, such as the bandwidth consumed when constantly sending an image via the Internet. Being able to have a service at the edge is beneficial because it avoids overloading the network with sending videos and an inference can be executed through a local network.
Considering that you can have multiple cameras, using a cloud service would be very expensive both from the internet connection, the cloud computing service and the cloud storage service. However, this does not mean that cloud services are not going to be required, but rather to optimize the sending of images since they can fill the bandwidth. The optimization would be carried out by sending only a few tables where particular information is found or by sending the statistical information. In this case, the duration and the number of people.

## Assess Model Use Cases

The application could have the following uses:
- Advertising a product: In commercial establishments you could place cameras and count how many people and how long it takes to see an advertisement or product on a counter shop. This would be useful to identify if the product or advertisement is attractive to customers. Additionally, you can identify which products are most attractive.


- Identification of people traffic in queues: It could be used to identify how many people are waiting for a service and how long it takes for the person to be served. In banks and supermarkets there are long queues at certain times. These could be avoided or optimized if the customer knew how many people there are before heading to the nearest supermarket or bank. In this situation where social distancing is important, the implementation of this idea is very useful.


- Transport distribution: Include cameras in transport vehicles and people at bus stops to determine the distribution of vehicles. Organize and synchronize it with traffic lights so that when there are no people, they work in favor of car traffic and when there are enough people they work in favor of human traffic. In the case of mass transport vehicles, determine the departures of the buses and trains according to the number of people waiting for the transport.

## Assess Effects on End User Needs

Both lighting, focal length and image size will depend on the system to be implemented. The images used to train the datasets have certain characteristics. In other words, it is not known how precise or how much precision a model will lose when using an image with low resolution or poor lighting. For this, the system would have to be evaluated and, if necessary, re-model the model (eg, through fine tunning) with the images from the camera to be used.
Therefore, it will be necessary to previously define at what distance the camera will be, the necessary resolution. These features could require cameras with higher costs, which would increase the cost of the system for the end user.
The precision of the results will depend on the model, there are models that may have better precision but require greater computational power. The solution should be tailored to what the end user needs. There is a compromise between computational power, required precision, and inference time. The minimum inference time will have to be defined with the end user and then the model to be used in the solution will be defined. This could increase costs, because if the end user needs a low latency and high precision solution, more expensive hardware will be required.

## Model Research


In investigating potential people counter models, I tried one additional model [YoloV3](), but since I had already worked with RCNN mask I preferred to use that model and the results were successful.

I Use the [Openvino guidelines](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html) for converting YOLO models.

I converted the model to an Intermediate Representation with the following arguments

```console
/opt/intel/openvino/deployment_tools/model_optimizer/mo.py  --input_model yolo_v3.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3.json --batch 1
```

The conversion was successful and I can even make an inference, I did not delve further as the problems I had with the Tensor Flow model to obtain the average accuracy were solved.


As a reference I used the [main.py]() for YOLOv3.