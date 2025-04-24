from ultralytics import YOLO

# Load a model
model = YOLO("home/tricubics/Desktop/Hand_Detection_In_Yolov8/yolov8_hand_detection/YOLOv8-pt/data/data.yaml")  # build a new model from YAML
model = YOLO("/home/tricubics/Desktop/Computer_Vision_TTS/Hand_Detection_In_Yolov8/yolov8_hand_detection/YOLOv8-pt/best_06.pt")  # load a pretrained model (recommended for training)
model = YOLO("home/tricubics/Desktop/Hand_Detection_In_Yolov8/yolov8_hand_detection/YOLOv8-pt/data/data.yaml").load("/home/tricubics/Desktop/Computer_Vision_TTS/Hand_Detection_In_Yolov8/yolov8_hand_detection/YOLOv8-pt/best_06.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="data.yaml", epochs=45, batch=32, imgsz=640)   # batch-size default = 64

