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


def parallel_laplacian_variance(file):
    """
    not mine. not used
    :param file:
    :return:
    """
    print(file + "             ", end="\r")
    img = cv2.imread(file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return laplacian


def frame_laplacian(frame):
    """
    not mine
    :param frame:
    :return:
    """
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return laplacian


def remove_blurry_frames(frames):
    """
    remove blurry frames from array of frames. no impact on stored video
    :param frames: array of frames
    :return: frames: kept frames
             output_dict: dictionary containing details of kept/removed frames
    """
    blurriness = np.array([frame_laplacian(frame) for frame in frames])
    median_blur = float(np.median(blurriness))
    # min_blur = float(np.min(blurriness))
    # max_blur = float(np.max(blurriness))
    adjusted_cutoff = 0.95 * median_blur
    # frames = np.array(frames)[median_blur]
    keep_idx = np.where(blurriness > adjusted_cutoff)
    remove_idx = np.where(blurriness <= adjusted_cutoff)
    frames = frames[keep_idx]

    output_dict = {"summary": f"kept {len(keep_idx[0])} of {len(keep_idx[0]) + len(remove_idx[0])} frames",
                   "kept frames": keep_idx[0].tolist(),
                   "removed frames": remove_idx[0].tolist()}
    return frames, output_dict


def remove_blurry_images(folder_name, extension, delete_frames=False):
    """
    removes blurry images from folder
    not currently used. refactor to take list of image arrays as input if to be used
    :param folder_name:
    :param extension:
    :param delete_frames:
    :return:
    """
    files = sorted(glob.glob(f'{folder_name}/*.{extension}'))
    # blurriness = np.zeros(len(files))
    print("Calculating Average Blurriness")

    # Parallelize
    # blurriness = [pool.apply(parallel_laplacian_variance, args=(file,)) for file in files]
    # pool.close()
    blurriness = [parallel_laplacian_variance(file) for file in files]
    median_blur = float(np.median(blurriness))
    min_blur = float(np.min(blurriness))
    max_blur = float(np.max(blurriness))
    print(median_blur)
    # wandb.log({'Individual Laplacian': blurriness, 'Batch Median Laplacian': median_blur})
    print("Median Blur (Laplacian Variance): " + str(median_blur))
    blur_cutoff = median_blur * 0.95  # + ((1-average_blur)*0.1)
    print("Blur Cutoff (Laplacian Variance): " + str(blur_cutoff))

    print("Removing Noisy Images")

    count = 0
    removed_files = []

    cutoffs = {}
    for percentage in np.arange(0, 1, 0.05):
        cutoffs.update({f'{percentage}': float(np.quantile(blurriness, percentage))})

    remainders = {}
    number_of_original_files = len(files)
    for j in np.arange(300, 1000, 100):
        remainder = sum(i > j for i in blurriness)
        remainders[f'{j} Threshold'] = float(remainder / number_of_original_files)

    remainders = dict(remainders)

    for i in range(len(files)):
        if blurriness[i] < blur_cutoff:
            # print("Deleting " + files[i] + " - Laplacian Noisiness: " + str(blurriness[i]))
            removed_files.append(files[i])
            if delete_frames == True:
                os.remove(files[i])
            count += 1
    blur_ratio = count / len(files)
    # wandb.log({'Noisy Frame Ratio': blur_ratio})
    print(f"Done Checking Frames, {count} frames removed.                 ")
    return {'Laplacian Threshold Remainders': remainders, 'Total Original Frames': number_of_original_files,
            'Removed Blurry Frame Count': count, 'Removed Blurry Frames': removed_files,
            'Median Laplacian Variance': median_blur, 'Minimum Laplacian Variance': min_blur,
            'Maximum Laplacian Variance': max_blur, 'Noisy Frame Ratio': blur_ratio, 'Laplacian Cutoffs': cutoffs}


def compare_images(i, files):
    """
    not mine. not used
    :param i:
    :param files:
    :return:
    """
    image1 = cv2.imread(files[i])
    image2 = cv2.imread(files[i + 1])
    image_gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image_gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    try:
        diff, _ = compare_ssim(image_gray1, image_gray2, full=True)
    except:
        image_gray2 = cv2.resize(image_gray2, image_gray1.shape, interpolation=cv2.INTER_AREA)
        diff, _ = compare_ssim(image_gray1, image_gray2, full=True)
    return diff





def remove_duplicates(folder_name, extension, delete_frames=False):
    """
    not mine. not used
    :param folder_name:
    :param extension:
    :param delete_frames:
    :return:
    """
    files = sorted(glob.glob(f'{folder_name}/*.{extension}'))
    print("Removing Duplicate and Highly Similar Frames\nCalculating Frame Similarities")
    # diff = np.zeros(len(files)-1)

    # Parallelize
    # pool = mp.Pool(mp.cpu_count())

    diff = [compare_images(i, files) for i in range(len(files) - 1)]

    # pool.close()

    median_diff = float(np.median(diff))
    # wandb.log({'Individual Frame Similarities': diff, 'Batch Median Frame Similarity': median_diff})

    diff_cutoff = median_diff * 1.05

    if diff_cutoff < 0.95:
        diff_cutoff = 0.95

    print(f'Similarity Cutoff (OpenCV Compare Images): {diff_cutoff}')
    print('Removing Duplicate Images')

    count = 0
    removed_files = []

    cutoffs = {}
    for percentage in np.arange(0, 1, 0.05):
        cutoffs.update({f'{percentage}': float(np.quantile(diff, percentage))})

    remainders = {}
    number_of_original_files = len(files)
    for j in np.arange(0.2, 0.8, 0.05):
        remainder = sum(i > j for i in diff)
        remainders[f'{j} Threshold'] = float(remainder / number_of_original_files)

    remainders = dict(remainders)

    for i in range(len(diff)):
        if diff[i] > 0.99:
            # print("Deleting " + files[i] + " - Similarity: " + str(diff[i]), end="\r")
            removed_files.append(files[i])
            if delete_frames:
                os.remove(files[i])
            # wandb.log({'Duplicates Similarity': diff})
            count += 1

    duplicate_ratio = count / len(files)
    # wandb.log({'Batch Duplicate Remove Ratio': duplicate_ratio})
    print("Done Checking Frames, " + str(count) + " frames removed.")
    return {'SSIM Threshold Remainders': remainders, 'Removed Duplicate Frame Count': count,
            'Removed Duplicate Frames': removed_files, 'Median Frame Similarity': median_diff,
            'Duplicate Frame Ratio': duplicate_ratio, 'Similarity Cutoffs': cutoffs}


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