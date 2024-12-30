# Vehicle_Recognition_System


A vehicle recognition system.

Usage image: 

$ python Main.py /LicPlateImages/1.png



optional arguments:
  -h, --help               show this help message and exit
  --yolo MODEL_PATH        path to YOLO model weight file, default yolo-coco
  --confidence CONFIDENCE  minimum probability to filter weak detections, default 0.5
  --threshold THRESHOLD    threshold when applying non-maxima suppression, default 0.3
  
Results: 

Before 
Example:
![](3.png)

After
![](imgOriginalScene.png)


Requirements
ultralytics>=8.3.31
ipywidgets>=8.1.5
python
numpy-1.26.4
pytesseract>=0.3.13
pandas>=2.2.2
matplotlib>=3.8.0
tensorflow
opencv-4.10.0
yolov3.weights must be downloaded from https://pjreddie.com/media/files/yolov3.weights and saved in folder yolo-coco





Licensed under the MIT License

