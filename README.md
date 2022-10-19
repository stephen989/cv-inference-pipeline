# cv-inference-pipeline

A computer vision inference pipeline for object detection that is designed for the YOLOv5 family of models. 

## Installation

`git clone https://github.com/stephen989/cv-inference-pipeline/

pip install -r requirements.txt`

## Usage for video
Prebuit weights available from pytorch:

`python main.py --video_input data/test.mp4 --video_output test_output.mp4 --yaml_output test_output.yaml --model yolov5 --model_version yolov5s --model_classes yolo_names.yaml`

Custom-trained weights:

`python main.py --video_input data/test.mp4 --video_output test_output.mp4 --yaml_output test_output.yaml --model yolov5 --weights path_to_weights --model_classes model_names.yaml`

This will output the location, class and confidence of each detection in the yaml file and create an output video which overlays the detected bounding boxes and class names on the input video.

## Usage for images
`python main.py --image_dir images --image_ext png --image_output_dir output_images --yaml_output images_output.yaml --model yolov5 --model_version yolov5s --model_classes yolo_names.yaml`

This will create a similar yaml file and a new directory containing the original images with a similar overlay as described above.
