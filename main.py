import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# Load pre-trained SSD MobileNet V2 model from TensorFlow Hub
print("Loading model...")
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

print("Model loaded.")

# COCO Labels
LABELS = [
    "???", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "???",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "???", "backpack",
    "umbrella", "???", "???", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "???", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "???", "dining table", "???",
    "???", "toilet", "???", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "???", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Start webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)
    tensor = tf.expand_dims(tensor, 0)

    # Run model
    result = model(tensor)
    result = {key: value.numpy() for key, value in result.items()}

    # Draw detections
    for i in range(len(result["detection_scores"][0])):
        score = result["detection_scores"][0][i]
        if score < 0.5:
            continue

        bbox = result["detection_boxes"][0][i]
        class_id = int(result["detection_classes"][0][i])

        label = LABELS[class_id] if class_id < len(LABELS) else "Unknown"

        h, w, _ = frame.shape
        y1, x1, y2, x2 = bbox
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Real-Time Object Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
