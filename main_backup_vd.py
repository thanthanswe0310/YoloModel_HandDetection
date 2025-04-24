import cv2
import numpy as np
import torch
import os
from datetime import datetime
import pytz
from ultralytics import YOLO


class TricubicsOpenStore:
    def __init__(self, input_source, model_path, output_path, min_distance_threshold=0.5, scale_factor=1.5):
        self.input_source = input_source  # Path to the video file
        self.model_path = model_path
        self.output_path = output_path
        self.min_distance_threshold = min_distance_threshold
        self.scale_factor = scale_factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device:', self.device)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

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

    def process_video_to_output(self):
        cap = cv2.VideoCapture(self.input_source)
        
        if not cap.isOpened():
            print(f"Error opening video file: {self.input_source}")
            return

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30  # Default to 30 if FPS not available

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
        output_video_path = os.path.join(self.output_path, 'output_video.mp4')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            results = self.model(frame)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()

                # Remove duplicate bounding boxes
                unique_boxes = self.remove_duplicates(boxes)

                # Draw the bounding boxes on the image
                for box in unique_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness 2

            # Write the processed frame to the video
            video_writer.write(frame)
            idx += 1

        cap.release()  # Release the video capture object
        video_writer.release()  # Finalize the video file
        print(f"Video processing completed. Output video saved at: {output_video_path}")


if __name__ == "__main__":
    current_datetime = datetime.now(pytz.timezone('Asia/Seoul'))

    output_dir = f"./captured_images/{current_datetime.strftime('%Y-%m-%d_%H-%M-%S')}/"
    input_source = "/home/tricubics/Desktop/Computer_Vision_TTS/Hand_Tracking/generated_videos/banana_uyu_left.mp4"  # Path to the video
    model_path = "/home/tricubics/Desktop/Computer_Vision_TTS/Hand_Detection_In_Yolov8/yolov8_hand_detection/YOLOv8-pt/best_latest_hand.pt"

    openstore = TricubicsOpenStore(input_source, model_path, output_dir)
    openstore.initialize_model()
    openstore.process_video_to_output()
