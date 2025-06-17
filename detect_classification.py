from ultralytics import YOLO
import cv2

# Load the object detection model
od_model = YOLO('/../Hand_Detection_In_Yolov8/yolov8_hand_detection/YOLOv8-pt/best_latest_hand.pt')

# Load the image classification model
ic_model =  YOLO('/../hand_and_tracking_test/best.pt')

# Load your image
image_path = '/../13_09_2024/KI1AEJ1000C7980734F00AEC3DF3E/video2/20240826_190311_091.jpg'

image = cv2.imread(image_path)

# Run object detection
od_results = od_model(image)

# Iterate over each detection
for det in od_results[0].boxes:
    # Extract bounding box coordinates
    x1, y1, x2, y2 = det.xyxy.cpu().numpy().astype(int)[0]
    # Crop the detected building from the original image
    crop_img = image[y1:y2, x1:x2]

    # Run image classification on the cropped image
    od_results = od_model(crop_img)
    print("IC results : ", od_results)

    # Check if classification result is not None and contains probabilities
    if od_results[0].probs is not None:
        class_label = od_results[0].probs.top1  # Get the top-1 class label
    else:
        class_label = "Unknown"  # Assign a default label if no classification result


## Here is an object detection
# Run object detection
ic_results = ic_model(image)

# Iterate over each detection
for det in ic_results[0].boxes:
    # Extract bounding box coordinates
    x1, y1, x2, y2 = det.xyxy.cpu().numpy().astype(int)[0]
    # Crop the detected building from the original image
    crop_ic_img = image[y1:y2, x1:x2]

    # Run image classification on the cropped image
    ic_results = ic_model(crop_ic_img)
    print("IC results : ", ic_results)

    # Check if classification result is not None and contains probabilities
    if ic_results[0].probs is not None:
        class_label = ic_results[0].probs.top1  # Get the top-1 class label
    else:
        class_label = "Unknown"  # Assign a default label if no classification result


    # Draw the bounding box and label on the original image
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, class_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Save or display the combined result
cv2.imwrite('/home/tricubics/Desktop/Computer_Vision_TTS/Hand_Detection_In_Yolov8/yolov8_hand_detection/YOLOv8-pt/combined_result.jpg', image)
