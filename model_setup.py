from .video_processing import *
import torch


class Model:
    def __init__(self, weights, version, model_classes_file):
        self.model, self.name = self.load_model(weights, version)
        if model_classes_file != "":
            with open(model_classes_file) as stream:
                self.classes = yaml.safe_load(stream)['names']
        else:
            self.classes = list(range(100))

    def load_model(self, weights_path, model_version):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if weights_path == "default":
            print("Loading default weights")
            model = torch.hub.load('ultralytics/yolov5', model_version).to(device)
        else:
            print("Loading custom weights")
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path).to(device)
        print("Model loaded")
        return model, model_version

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        results_device = self.model(img)  # batch of images
        results = results_device.pred[0].to('cpu').numpy()

        coords = results[:, :4]
        coords = coords.reshape(coords.shape[:-1] + (2, 2)).astype("int")
        coords = coords.tolist()
        prediction_scores = results[:, 4].round(3).tolist()
        prediction_classes = results[:, 5].astype("int").tolist()

        output_dict = {"detection_boxes": coords,
                       "detection_classes": prediction_classes,
                       "detection_scores": prediction_scores,
                       "num_detections": len(coords)}
        return output_dict
