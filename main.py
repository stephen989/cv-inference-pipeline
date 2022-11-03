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


def video_pipeline(
        video_input='videos/test.mp4',
        video_output='oil_output.mp4',
        yaml_output='test_pipeline.yaml',
        model_version='yolov5s',
        weights='default',
        model_classes=''):
    """
    See function: main for docs
    """
    video = video_input
    output_yaml=yaml_output
    video_preproc = VideoPreprocessor()
    frames, FPS = video_preproc.preprocess_video(video)
    outputs_dict = {"File": video,
                    "Output Video": video_output,
                    "Model Outputs": dict()}

    model = Model(weights, model_version, model_classes)
    outputs_dict["Model classes"] = model.classes

    # feed through model one by one
    print("Feeding model")
    model_outputs = list()
    for i, frame in enumerate(tqdm(frames, unit="frame")):
        output = model(frame)
        model_outputs.append(output)
        outputs_dict["Model Outputs"][i] = output
    print("Complete")
    create_output_video(frames, model_outputs, video_output, FPS)
    print(f"Saved output video to {video_output}")
    return True





def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='minidata/test/images')
    parser.add_argument('--image_ext', type=str, default='.jpg')
    parser.add_argument('--image_output_dir', type=str, default='output_imagesbad')
    parser.add_argument('--video_input', type=str, default='data/test.mp4')
    parser.add_argument('--video_output', type=str, default='oil_output.mp4')
    parser.add_argument('--yaml_output', type=str, default='test_pipeline.yaml')
    parser.add_argument('--model_version', type=str, default='yolov5s')
    parser.add_argument('--weights', type=str, default='default')
    parser.add_argument('--model_classes', type=str, default='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(image_dir='minidata/test/images',
        image_ext='.jpg',
        image_output_dir='output_imagesbad',
        video_input='data/test.mp4',
        video_output='oil_output.mp4',
        yaml_output='test_pipeline.yaml',
        model_version='yolov5s',
        weights='default',
        model_classes=''):
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
        video_pipeline(image_dir,
                        image_ext,
                        image_output_dir,
                        video_input,
                        video_output,
                        yaml_output,
                        model_version,
                        weights,
                        model_classes)
    else:
        image_pipeline(image_dir,
                        image_ext,
                        image_output_dir,
                        yaml_output,
                        model_version,
                        weights,
                        model_classes)
    return True

if __name__ == "__main__":
    
    os.chdir("..")
    main(parse_opt())
