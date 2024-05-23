import cv2
import numpy as np
import torch
import math
import time
import random
import os
from ultralytics import YOLO
from sort import *
import cvzone

# define display parameters
font = cv2.FONT_HERSHEY_PLAIN
font_scale = 2
font_thickness = 2
rect_thickness = 2
obj_fill = cv2.FILLED
text_color = (255, 255, 255)
process_text_color = (255, 255, 255)
rect_color = (255, 0, 255)
background_color = (255, 0, 255)
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(20)]

# define processing device mode
device = "mps" if torch.backends.mps.is_available() else "cpu"
# define yolo weight
model_path = os.path.join(".", "weights", "yolov8x.pt")
model = YOLO(model_path)
model.to(device)

# define video resolution
WIDTH, HEIGHT = 1280, 720

# define video path
video_path = os.path.join(".", "videos", "sheep.mp4")
output_path = os.path.join(".", "videos", "output.mp4")

# define video operation instance
cap = cv2.VideoCapture(video_path)
cap_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MP4V"), cap.get(cv2.CAP_PROP_FPS), (WIDTH, HEIGHT))

# set resolution
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# mouse callback function
def show_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinates: x={x}, y={y}")
# define window name
cv2.namedWindow("Frame")
# set the mouse callback function
cv2.setMouseCallback("Frame", show_coordinates)

# define time to calculate FPS
previous_time = time.time()

# define ROI mask
mask = cv2.imread("./images/sheep_mask.png")

# define tracking instance
tracker = Sort(max_age=99, min_hits=3, iou_threshold=0.3)

# define counter line
counterLine = [430, 567, 1276, 206]
totalCount = set()

def line_eq(x, line):
    """Calculate the y value on the line at x"""
    x1, y1, x2, y2 = line
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

def is_crossing(cx, cy, prev_cx, prev_cy, line):
    """Check if the line between (prev_cx, prev_cy) and (cx, cy) crosses the given line"""
    x1, y1, x2, y2 = line
    # If both points are on the same side of the line, there's no crossing
    if (cy > line_eq(cx, line)) == (prev_cy > line_eq(prev_cx, line)):
        return False
    return True

# Track previous center points
prev_centers = {}

# couter background
counter_overlay_path = os.path.join(".", "images", "sheep_counter_bg.png")
counter_overlay = cv2.imread(counter_overlay_path, cv2.IMREAD_UNCHANGED)
counter_overlay = cv2.resize(counter_overlay, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)

while True:
    success, frame = cap.read()
    
    if not success:
        break
    
    cvzone.overlayPNG(frame, counter_overlay, (120, frame.shape[0] - 200))
    
    results = model(frame, stream=True)
    
    detections = np.empty((0, 5))
    
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    
    # cv2.putText(frame, f"[Processing Mode: {device}] [FPS: {int(fps)}]", (20, frame.shape[0] - 20), font, font_scale, process_text_color, font_thickness, lineType=cv2.LINE_AA)
    
    for r in results:
        classNames = r.names   
        boxes = r.boxes
        
        for bbox in boxes:
            x1, y1, x2, y2 = bbox.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cls = int(bbox.cls[0])
            currentClass = classNames[cls]
            
            conf = math.ceil(bbox.conf[0] * 100) / 100
            
            if currentClass == "sheep" and conf > 0.5:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                
        resultsTracker = tracker.update(detections)
        
        cv2.line(frame, (counterLine[0], counterLine[1]), (counterLine[2], counterLine[3]), (0, 0, 255), 4)
        
        for result in resultsTracker:
            x1, y1, x2, y2, Id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            
            if Id in prev_centers:
                prev_cx, prev_cy = prev_centers[Id]
                
                if is_crossing(cx, cy, prev_cx, prev_cy, counterLine):
                    if Id not in totalCount:
                        totalCount.add(Id)
                        cv2.line(frame, (counterLine[0], counterLine[1]), (counterLine[2], counterLine[3]), (0, 255, 0), 3)
            
            prev_centers[Id] = (cx, cy)
            
            cv2.putText(frame, f"{len(totalCount)}", (250, frame.shape[0] - 120), font, font_scale + 2, (0, 0, 0), font_thickness + 2, lineType=cv2.LINE_AA)
            
            currentId = int(Id)
            text = f"{currentClass} ID: {currentId}"
             
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[currentId % len(colors)], rect_thickness)
            
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            x2, y2 = x1 + text_width, y1 - text_height - baseline
            
            cv2.rectangle(frame, (max(0, x1), max(35, y1)), (x2, y2), colors[currentId % len(colors)], obj_fill)
            cv2.putText(frame, text, (max(0, x1), max(35, y1)), font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 5, colors[currentId % len(colors)], obj_fill)
            
        cv2.imshow("Frame", frame)
        cap_out.write(frame)
        
        key = cv2.waitKey(1)
        if key == 27:
            break
        
cap.release()
cap_out.release()
cv2.destroyAllWindows()