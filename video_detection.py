from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

def detect_objs(frame):
    detections = model(frame)

    detected_objs = []
    for detection in detections:
        boxes = detection.boxes.xyxy
        confidence = detection.boxes.conf
        class_ids = detection.boxes.cls

        detected_objs.append([boxes,confidence,class_ids])
        
        return detected_objs
    
def process_detected_objs(detected_objs,frame):
    for boxes,confidence,class_ids in detected_objs:
        count = sum(class_id == 0 for class_id in class_ids)
        person_count = int(count)
        print("Number of People detected:",person_count)

        for box,conf,class_id in zip(boxes,confidence,class_ids):
            x1,y1,x2,y2 = map(int,box)
            label = f"Class {class_id.item()},confidence {conf:.2f}"
            color = (0,255,0)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)


def detector(source=0):
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        success,frame = cap.read()
        if not  success:
            print('camera not opened')
            break
        color_change = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        detected_objs = detect_objs(color_change)
        process_detected_objs(detected_objs,frame)
        # detection_plot = detection.plot()
        cv2.imshow("Object detection",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    detector()