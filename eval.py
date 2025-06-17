from ultralytics import YOLO

# Load a pretrained model
model = YOLO(r'/../Hand_Detection_In_Yolov8/yolov8_hand_detection/YOLOv8-pt/best_21_09_2024.pt') 

# Evaluate the model on a validation dataset
results = model.train(data=r'/../Hand_Detection_In_Yolov8/yolov8_hand_detection/YOLOv8-pt/data/data.yaml',epochs=2)

# Print the entire results to inspect the available attributes
print(results)

# Access individual metrics
map50 = results.box.map50    # mAP@0.5
map5095 = results.box.map    # mAP@0.5:0.95
precision = results.box.maps[0]  # Precision for the first class (can average across classes if needed)
recall = results.box.maps[1]  # Recall for the first class (can average across classes if needed)


# Calculate F1-score
f1_score = 2 * (precision * recall) / (precision + recall) 


print(f"mAP@0.5: {map50:.4f}")
print(f"mAP@0.5:0.95: {map5095:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score (as a form of accuracy): {f1_score:.3f}") 


# import os
# import pandas as pd

# def count_hands(yolo_labels_dir):
#     left_hand_count = 0
#     right_hand_count = 0

#     # Iterate over all label files in the directory
#     for filename in os.listdir(yolo_labels_dir):
#         if filename.endswith('.txt'):  # Assuming YOLOv8 labels are in .txt files
#             label_path = os.path.join(yolo_labels_dir, filename)
            
#             # Read the label file
#             with open(label_path, 'r') as file:
#                 for line in file:
#                     # Assuming each line follows the YOLO format: class_id x_center y_center width height
#                     values = line.strip().split()
#                     class_id = int(values[0])

#                     if class_id == 0:  # Assuming class_id 0 is for left hand
#                         left_hand_count += 1
#                     elif class_id == 1:  # Assuming class_id 1 is for right hand
#                         right_hand_count += 1

#     return left_hand_count, right_hand_count

# # Directory containing YOLOv8 label files
# yolo_labels_dir = r'/../Hand_Detection_In_Yolov8/yolov8_hand_detection/YOLOv8-pt/data/train/labels'

# left_hand_count, right_hand_count = count_hands(yolo_labels_dir)

# print(f"Number of left hands detected: {left_hand_count}")
# print(f"Number of right hands detected: {right_hand_count}")
