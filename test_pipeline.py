import argparse
from model_setup import *
os.chdir("C:\\Users\\RW154JK\\OneDrive - EY\\Desktop\\Kerry")


def test_preproc(video):
    """
    :param video: video location
    :return: array of processed frames
    """
    print("Preprocessing video")
    # split into frames
    frames = split_video(video)
    # remove blurry
    frames, blurry_output = remove_blurry_frames(frames)
    # remove duplicates
    pass

    outputs = [blurry_output]
    print("Complete")
    return frames, outputs


def test_pipeline(video,
                  output_video,
                  output_yaml,
                  load_model_fn,
                  draw=False):
    """
    :param video: input video file path
    :param output_video: output video file path
    :param output_yaml: output yaml file path
    :param load_model_fn: function to load and return model to be used
    :param draw: if true, draw on image
    :return: True if success
    """

    # preprocessing - split into frame, remove blurry and similar frames ???
    frames, outputs = test_preproc(video)
    # load model
    model = Model(load_model_fn)
    # feed through model one by one
    outputs_dict = {"File": video,
                    "Output Video": output_video,
                    "Model": model.type,
                    "Preprocessing output": outputs,
                    "Model Outputs": dict()}
    print("Feeding model")
    for i, frame in enumerate(frames):
        output = model(frame)
        outputs_dict["Model Outputs"][i] = output
    print("Complete")
    # write to yaml file
    with open(output_yaml, 'w') as y:
        yaml.dump(outputs_dict, y)

    if draw:
        create_output_video(output_yaml, frames, output_video, video)

    return True

def main(opts):
    """
    :param video: input video file path
    :param output_video: output video file path
    :param output_yaml: output yaml file path
    :param load_model_fn: function to load and return model to be used
    :param draw: if true, draw on image
    :return: True if success
    """

    video = opts.video_input
    output_video = opts.video_output
    output_yaml = opts.yaml_output
    model = opts.model
    weights = opts.weights

    # preprocessing - split into frame, remove blurry and similar frames ???
    frames, outputs = test_preproc(video)
    # load model
    model = Model(model, weights, opts.model_version)
    # feed through model one by one
    outputs_dict = {"File": video,
                    "Output Video": output_video,
                    "Model": model.type,
                    "Preprocessing output": outputs,
                    "Model Outputs": dict()}
    print("Feeding model")
    for i, frame in enumerate(frames):
        output = model(frame)
        outputs_dict["Model Outputs"][i] = output
    print("Complete")
    # write to yaml file
    with open(output_yaml, 'w') as y:
        yaml.dump(outputs_dict, y)

    create_output_video(output_yaml, frames, output_video, video)

    return True





def parse_opt():

    parser = argparse.ArgumentParser()
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
