import argparse
import os
from model_setup import *
os.chdir("..")


def image_main(opts):
    image_dir = opts.image_dir
    image_ext = opts.image_ext
    image_output_dir = opts.image_output_dir
    video = opts.video_input
    output_video = opts.video_output
    output_yaml = opts.yaml_output
    model = opts.model
    weights = opts.weights

    os.makedirs(image_output_dir, exist_ok=True)


    # feed through model one by one
    frames, outputs, image_names = image_preprocessing(image_dir, image_ext)
    outputs_dict = {"Directory": image_dir,
                    "Extension": image_ext,
                    "Output directory": image_output_dir,
                    "Model": model,
                    "Preprocessing output": outputs,
                    "Model Outputs": dict()}

    model = Model(model, weights, opts.model_version)

    print("Feeding model")
    for image_name, frame in zip(image_names, frames):
        output = model(frame)
        outputs_dict["Model Outputs"][image_name] = output
    print("Complete")
    # write to yaml file
    with open(output_yaml, 'w') as y:
        yaml.dump(outputs_dict, y)


    create_output_images(output_yaml, frames,  image_output_dir)



def video_main(opts):
    image_dir = opts.image_dir
    image_ext = opts.image_ext
    image_output_dir = opts.image_output_dir
    video = opts.video_input
    output_video = opts.video_output
    output_yaml = opts.yaml_output
    model = opts.model
    weights = opts.weights

    frames, outputs = video_prepocessing(video)
    outputs_dict = {"File": video,
                    "Output Video": output_video,
                    "Model": model,
                    "Preprocessing output": outputs,
                    "Model Outputs": dict()}

    model = Model(model, weights, opts.model_version)
    # feed through model one by one

    print("Feeding model")
    for i, frame in enumerate(frames):
        output = model(frame)
        outputs_dict["Model Outputs"][i] = output
    print("Complete")
    # write to yaml file
    with open(output_yaml, 'w') as y:
        yaml.dump(outputs_dict, y)
    create_output_video(output_yaml, frames, output_video, video)


def main(opts):
    """
    :param video: input video file path
    :param output_video: output video file path
    :param output_yaml: output yaml file path
    :param load_model_fn: function to load and return model to be used
    :param draw: if true, draw on image
    :return: True if success
    """
    image_dir = opts.image_dir
    image_ext = opts.image_ext
    image_output_dir = opts.image_output_dir
    video = opts.video_input
    output_video = opts.video_output
    output_yaml = opts.yaml_output
    model = opts.model
    weights = opts.weights

    # preprocessing - split into frame, remove blurry and similar frames ???
    if video != '':
        video_main(opts)
    else:
        image_main(opts)
    return True








def parse_opt(known=False):

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='minidata/test/images')
    parser.add_argument('--image_ext', type=str, default='.jpg')
    parser.add_argument('--image_output_dir', type=str, default='output_images')
    parser.add_argument('--video_input', type=str, default='')
    parser.add_argument('--video_output', type=str, default='test_output.mp4')
    parser.add_argument('--yaml_output', type=str, default='test_pipeline.yaml')
    parser.add_argument('--model', type=str, default='yolo')
    parser.add_argument('--model_version', type=str, default='yolov5s')
    parser.add_argument('--weights', type=str, default='default')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    # test_pipeline("test.mp4",
    #               "test_output.mp4",
    #               "test_pipeline.yaml",
    #               load_yolo,
    #               True
    #               )
    opts = parse_opt()
    main(opts)
