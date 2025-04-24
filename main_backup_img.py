# import cv2
# import numpy as np
# import torch
# import os
# from datetime import datetime
# import pytz
# from ultralytics import YOLO

# class TricubicsOpenStore:
#     def __init__(self, input_source, model_path, output_path, min_distance_threshold=0.5, scale_factor=1.5):
#         self.input_source = input_source  # Directory containing images
#         self.model_path = model_path
#         self.output_path = output_path
#         self.min_distance_threshold = min_distance_threshold
#         self.scale_factor = scale_factor
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print('Device:', self.device)

#         if not os.path.exists(self.output_path):
#             os.makedirs(self.output_path)

#         self.crop_dir_name = self.output_path
#         if not os.path.exists(self.crop_dir_name):
#             os.mkdir(self.crop_dir_name)

#         self.iou_threshold = 0.1

#     @torch.no_grad()
#     def initialize_model(self):
#         self.model = YOLO(self.model_path).to(self.device)
#         n = torch.cuda.device_count()
#         if n > 1:
#             print(f"Using {n} GPUs with DataParallel")
#             self.model = torch.nn.DataParallel(self.model)
#         print(f"Model initialized on device: {self.device}")

#     def get_IOU(self, box1, box2):
#         x1, y1, x2, y2 = box1
#         x3, y3, x4, y4 = box2
#         inter_x1 = max(x1, x3)
#         inter_y1 = max(y1, y3)
#         inter_x2 = min(x2, x4)
#         inter_y2 = min(y2, y4)

#         if inter_x2 < inter_x1 or inter_y2 < inter_y1:
#             return 0.0

#         inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
#         box1_area = (x2 - x1) * (y2 - y1)
#         box2_area = (x4 - x3) * (y4 - y3)

#         union_area = box1_area + box2_area - inter_area

#         return inter_area / union_area

#     def remove_duplicates(self, boxes):
#         unique_boxes = []
#         for i, box in enumerate(boxes):
#             is_duplicate = False
#             for j, other_box in enumerate(boxes):
#                 if i != j and self.get_IOU(box, other_box) > self.iou_threshold:
#                     is_duplicate = True
#                     break
#             if not is_duplicate:
#                 unique_boxes.append(box)
#         return unique_boxes

#     def process_images(self):
#         image_files = [f for f in os.listdir(self.input_source) if f.endswith(('.png', '.jpg', '.jpeg'))]
#         image_files.sort()  # Sort files if needed
#         idx = 0

#         for image_file in image_files:
#             image_path = os.path.join(self.input_source, image_file)
#             frame = cv2.imread(image_path)
#             if frame is None:
#                 print(f"Error reading image {image_file}")
#                 continue

#             result = self.model(frame)[0]  # Corrected model call (for single GPU or multi-GPU)
#             boxes = result.boxes.xyxy.cpu().numpy()

#             # Remove duplicate bounding boxes
#             unique_boxes = self.remove_duplicates(boxes)

#             # Resize the entire image to 1920x1080 and save it
#             resized_frame = cv2.resize(frame, (1920, 1080))

#             idx += 1
#             cv2.imwrite(os.path.join(self.crop_dir_name, f"{idx}.png"), resized_frame)

#         print("Image processing completed.")

# if __name__ == "__main__":

#     current_datetime = datetime.now(pytz.timezone('Asia/Seoul'))

#     output_dir = f"./captured_images/{current_datetime.strftime('%Y-%m-%d_%H-%M-%S')}/"
#     input_source = "/home/tricubics/Desktop/13_09_2024/KI1AEJ1000C7980734F00AEC3DF3E/video2"  # Directory containing the images
#     model_path = "/home/tricubics/Desktop/Computer_Vision_TTS/Hand_Detection_In_Yolov8/yolov8_hand_detection/YOLOv8-pt/best_21_09_2024.pt"

#     openstore = TricubicsOpenStore(input_source, model_path, output_dir)
#     openstore.initialize_model()
#     openstore.process_images()



import cv2
import numpy as np
import torch
import os
from datetime import datetime
import pytz
from ultralytics import YOLO

class TricubicsOpenStore:
    def __init__(self, input_source, model_path, output_path, min_distance_threshold=0.5, scale_factor=1.5):
        self.input_source = input_source  # Directory containing images
        self.model_path = model_path
        self.output_path = output_path
        self.min_distance_threshold = min_distance_threshold
        self.scale_factor = scale_factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device:', self.device)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.crop_dir_name = self.output_path
        if not os.path.exists(self.crop_dir_name):
            os.mkdir(self.crop_dir_name)

        self.iou_threshold = 0.1

    @torch.no_grad()
    def initialize_model(self):
        self.model = YOLO(self.model_path).to(self.device)
        n = torch.cuda.device_count()
        if n > 1:
            print(f"Using {n} GPUs with DataParallel")
            self.model = torch.nn.DataParallel(self.model)
        print(f"Model initialized on device: {self.device}")

    def get_IOU(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        inter_x1 = max(x1, x3)
        inter_y1 = max(y1, y3)
        inter_x2 = min(x2, x4)
        inter_y2 = min(y2, y4)

        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)

        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area

    def remove_duplicates(self, boxes):
        unique_boxes = []
        for i, box in enumerate(boxes):
            is_duplicate = False
            for j, other_box in enumerate(boxes):
                if i != j and self.get_IOU(box, other_box) > self.iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_boxes.append(box)
        return unique_boxes

    def process_images(self):
        image_files = [f for f in os.listdir(self.input_source) if f.endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()  # Sort files if needed
        idx = 0

        for image_file in image_files:
            image_path = os.path.join(self.input_source, image_file)
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Error reading image {image_file}")
                continue

            results = self.model(frame)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()

                # Remove duplicate bounding boxes
                unique_boxes = self.remove_duplicates(boxes)

                # Draw the bounding boxes on the image
                for box in unique_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness 2

                # Resize the entire image to 1920x1080 and save it
                resized_frame = cv2.resize(frame, (1920, 1080))
                idx += 1
                cv2.imwrite(os.path.join(self.crop_dir_name, f"{idx}.png"), resized_frame)

        print("Image processing completed.")

if __name__ == "__main__":
    current_datetime = datetime.now(pytz.timezone('Asia/Seoul'))

    output_dir = f"./captured_images/{current_datetime.strftime('%Y-%m-%d_%H-%M-%S')}/"
    input_source = "/home/tricubics/Desktop/13_09_2024/KI1AEJ1000C7980734F00AEC3DF3E/video2"  # Directory containing the images
    model_path = "/home/tricubics/Desktop/Computer_Vision_TTS/Hand_Detection_In_Yolov8/yolov8_hand_detection/YOLOv8-pt/best_21_09_2024.pt"

    openstore = TricubicsOpenStore(input_source, model_path, output_dir)
    openstore.initialize_model()
    openstore.process_images()
