from video_processing import *

CLASS_LABELS = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck",
                9: "boat", 10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
                16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear",
                24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
                34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
                40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle",
                46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana",
                53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza",
                60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 67: "dining table",
                70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone",
                78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock",
                86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush"}

model_url = 'http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz'
base_url = os.path.dirname(model_url) + "/"
model_file = os.path.basename(model_url)
model_name = os.path.splitext(os.path.splitext(model_file)[0])[0]
model_dir = tf.keras.utils.get_file(fname=model_name, origin=base_url + model_file, untar=True)
model_dir = pathlib.Path(model_dir) / "saved_model"
model = tf.saved_model.load(str(model_dir))
model = model.signatures['serving_default']


def detect_objects(image):
    img = Image.open(image)
    input_tensor = tf.convert_to_tensor(img)
    input_tensor = input_tensor[tf.newaxis, ...]
    output_dict = model(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
    classes = output_dict['detection_classes'].astype(np.int64)
    class_names = [None] * len(classes)
    for i in range(len(classes)):
        class_names[i] = CLASS_LABELS[classes[i]]
    output = Counter(class_names)
    # wandb.log({'Detections per Image': num_detections})
    return {'Objects Detected': output, 'Number of Detections': num_detections}

def detect_file(folder_name, extension):
    detect_list = []
    for file in sorted(glob.glob(f'{folder_name}/*.{extension}')):
        print(f'Detecting on Image: {file}', end='\r')
        output = {'Frame': file}
        output.update(detect_objects(file))
        detect_list.append(output)
    return {'Classification Information': detect_list}