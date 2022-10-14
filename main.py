from test_pipeline import *
import argparse

os.chdir("..")


def main(opts):
    """
    :param opts: command line options input
    image_dir: directory of input images (if using image pipeline)
    img_ext: file extenstion of input images
    image_output_dir: output_directory for images
    video_input: video input file if using video pipeline
    video_output: output location for video file
    yaml_output: name/location for yaml output to be saved
    model: type of model to be used. one of yolo or resnet (pls use yolo)
    model_version: which model version to use if choosing yolo
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
    parser.add_argument('--model', type=str, default='yolo')
    parser.add_argument('--model_version', type=str, default='yolov5s')
    parser.add_argument('--weights', type=str, default='default')
    parser.add_argument('--model_classes', type=str, default='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opts = parse_opt()
    main(opts)
