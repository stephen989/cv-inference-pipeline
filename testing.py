import os
import numpy as np
import cv2
import yaml

os.chdir("C:\\Users\\RW154JK\\OneDrive - EY\\Desktop\\Kerry")
with open("test_pipeline.yaml") as stream:
    outputs = yaml.safe_load(stream)

boxes = np.array(outputs["Model Outputs"][0]['detection_boxes'])
video_capture = cv2.VideoCapture("test.mp4")
success, frame = video_capture.read()



def draw_boxes(frame, boxes):
    for (p1, p2) in boxes:

        cv2.rectangle(frame, tuple(p1), tuple(p2), color=(0, 0, 255))
        while True:
            cv2.imshow("Video feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    return frame



draw_boxes(frame, boxes)
print(boxes)

