from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
from glob import glob
import yaml
import cv2
import os
from PIL import Image


class ImagePreprocessor:
    def __init__(self):
        self.params = {}

    def preprocess_images(self, image_dir, image_ext):
        """
        get all images from folder and return them with output dict and list of image file names
        for saving model outputs. Currently does not edit/remove/deblur images
        :param image_dir:
        :param image_ext:
        :return:
        """
        image_names = glob(f"{image_dir}/*{image_ext}")
        images_list = np.array([cv2.imread(file) for file in image_names], dtype=object)
        if len(images_list) == 0:
            raise ValueError(f"No {image_ext} files found in {image_dir}")
        images = []
        for image in images_list:
            try:
                images.append(cv2.cvtColor(np.array(image, dtype='uint8'), cv2.COLOR_BGR2RGB))
            except Exception as e:
                print(f"Unable to read {image}.\n{str(e)}")
        if len(images) == 0:
            raise ValueError(f"unable to read images from {image_dir}/*{image_ext}")
        outputs = []
        return images, outputs, [image_name.split("\\")[-1] for image_name in image_names]


class VideoPreprocessor(ImagePreprocessor):
    def __int__(self):
        self.params = {}

    def preprocess_video(self, video):
        """
        :param video: video location
        :return: array of processed frames
        """
        print("Preprocessing video")
        # split into frames
        frames, fps = self.split_video(video)
        # remove blurry
        # frames, blurry_output = remove_blurry_frames(frames)
        # outputs = [blurry_output]
        print("Complete")
        return frames[:3], fps

    def split_video(self, video):
        """
        splits video file into individual frames and returns them as array
        :param video: location
        :return: array of frames
        """
        frames = []
        video_capture = cv2.VideoCapture(video)
        success, frame = video_capture.read()
        while success:
            frames.append(frame)
            success, frame = video_capture.read()
        if len(frames) == 0:
            raise ValueError(f"Unable to retrieve any frames from {video}.")
        return np.array(frames), video_capture.get(cv2.CAP_PROP_FPS)











def draw_boxes(frame, model_output):
    """
    function to draw boxes around detected objects and label them with class
    :param frame: image as array
    :param model_output: [[x1, y1, x2, y2, class, conf]]
    :return: annotated frame as array
    """

    for (x1, y1, x2, y2, clss, conf) in model_output:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255))
        # cv2.putText(frame, str(int(id)), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return frame


def create_video_writer(video_width, video_height, fps, output_path):
    """
    create and return video writes
    :param video_width:
    :param video_height:
    :param fps: input video source (for fps)
    :param output_path:
    :return: video writer
    """
    # Getting the fps of the source video
    # initialize our video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps,
                           (video_width, video_height), True)


def create_output_video(frames, model_outputs, output_video, fps):
    """
    create output video with annotations and labels
    :param yaml_file: model output file name
    :param frames: frames which were fed to model
    :param output_video: video destination file
    :param video: original input video
    :return: None
    """
    video_height, video_width, _ = frames[0].shape
    writer = create_video_writer(video_width, video_height, fps, output_video)
    for i, frame in enumerate(frames):
        frame = draw_boxes(frame, model_outputs[i])
        writer.write(frame)
    writer.release()
    return True


def create_output_images(yaml_file, frames, image_output_dir):
    """
    create output images with labels and annotations
    :param yaml_file: model output file name
    :param frames: list of original frames
    :param image_output_dir: destination
    :return:
    """
    print(yaml_file)
    with open(yaml_file) as stream:
        outputs_dict = yaml.safe_load(stream)
    image_names = outputs_dict["Model Outputs"].keys()
    model_classes = outputs_dict['Model classes']
    for image_name, frame in zip(image_names, frames):
        frame = draw_boxes(frame,
                           outputs_dict["Model Outputs"][image_name],
                           model_classes)
        img = Image.fromarray(frame)
        img_name = f"{outputs_dict['Output directory']}/{image_name}.{outputs_dict['Extension']}"
        img.save(img_name)
    return True