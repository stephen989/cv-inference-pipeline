import argparse
from .model_setup import Model
from tqdm import tqdm
import os
from .model_setup import *



def image_pipeline(opts):
    """
    See function: main for docs
    """

    image_dir = opts.image_dir
    image_ext = opts.image_ext
    image_output_dir = opts.image_output_dir
    output_yaml = opts.yaml_output
    weights = opts.weights
    model_classes_file = opts.model_classes

    os.makedirs(image_output_dir, exist_ok=True)
    image_preproc = ImagePreprocessor()
    frames, outputs, image_names = image_preproc.preprocess_images(image_dir, image_ext)
    outputs_dict = {"Directory": image_dir,
                    "Extension": image_ext,
                    "Output directory": image_output_dir,
                    "Preprocessing output": outputs,
                    "Model Outputs": dict()}

    model = Model(weights, opts.model_version, model_classes_file)
    outputs_dict["Model classes"] = model.classes
    print("Feeding model")
    for image_name, frame in tqdm(zip(image_names, frames), total=len(frames)):
        output = model(frame)
        outputs_dict["Model Outputs"][image_name] = output
    print("Complete")
    # write to yaml file
    with open(output_yaml, 'w') as y:
        yaml.dump(outputs_dict, y)
    print(f"Saved run output to {output_yaml}")
    create_output_images(output_yaml, frames,  image_output_dir)
    print(f"Saved output images to {image_output_dir}")
    return True


def video_pipeline(opts):
    """
    See function: main for docs
    """
    video = opts.video_input
    output_video = opts.video_output
    output_yaml = opts.yaml_output
    weights = opts.weights
    model_classes = opts.model_classes
    video_preproc = VideoPreprocessor()
    frames, outputs = video_preproc.preprocess_video(video)
    outputs_dict = {"File": video,
                    "Output Video": output_video,
                    "Preprocessing output": outputs,
                    "Model Outputs": dict()}

    model = Model(weights, opts.model_version, model_classes)
    outputs_dict["Model classes"] = model.classes

    # feed through model one by one
    print("Feeding model")
    for i, frame in enumerate(tqdm(frames, unit="frame")):
        output = model(frame)
        outputs_dict["Model Outputs"][i] = output
    print("Complete")
    # write to yaml file
    with open(output_yaml, 'w') as y:
        yaml.dump(outputs_dict, y)
    print(f"Saved run output to {output_yaml}")
    create_output_video(output_yaml, frames, output_video, video)
    print(f"Saved output video to {output_video}")
    return True


def main(opts):
    """
    :param opts: command line options input
    image_dir: directory of input images (if using image pipeline)
    img_ext: file extenstion of input images
    image_output_dir: output_directory for images
    video_input: video input file if using video pipeline
    video_output: output location for video file
    yaml_output: name/location for yaml output to be saved
    model_version: which model version to use
    weights: location of saved weights to use if not loading default model
    model_classes: name/location of yaml file which maps model classes to numbers in "names"
    """
    video = opts.video_input
    if video != '':
        video_pipeline(opts)
    else:
        image_pipeline(opts)
    return True


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='minidata/test/images')
    parser.add_argument('--image_ext', type=str, default='.jpg')
    parser.add_argument('--image_output_dir', type=str, default='output_imagesbad')
    parser.add_argument('--video_input', type=str, default='')
    parser.add_argument('--video_output', type=str, default='oil_output.mp4')
    parser.add_argument('--yaml_output', type=str, default='test_pipeline.yaml')
    parser.add_argument('--model_version', type=str, default='yolov5s')
    parser.add_argument('--weights', type=str, default='default')
    parser.add_argument('--model_classes', type=str, default='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    
    os.chdir("..")
    options = parse_opt()
    main(options)
