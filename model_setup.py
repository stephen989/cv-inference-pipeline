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
        print(output_dict["detection_boxes"])

        # [[x1, y1], [x2, y2]] to [x1, y1, x2, y2]
        x = np.array(output_dict["detection_boxes"])
        x= x.reshape(len(x),-1)
        print(x)
        
        confs = np.expand_dims(np.array(output_dict['detection_scores']), 1)
        clss = np.expand_dims(np.array(output_dict['detection_classes']), 1)
        print(np.hstack([x, confs, clss]))
        return np.hstack([x, confs, clss])

    def xywh_convert(model_output):
            x = model_output['detection_boxes']
            x = np.array(x)
            y = np.zeros((x.shape[0], 4))
            y[:,:2] = x.mean(1)
            y[:,2] = x[:,1,0] - x[:,0,0]
            y[:,3] = x[:,1,1] - x[:,0,1]
            y = np.abs(y)
            return y
