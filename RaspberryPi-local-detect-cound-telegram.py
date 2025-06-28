from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np
import cv2
import requests
import time  # Added for frame rate control

# Telegram bot parameters
TOKEN = "7290187905:AAHp7vnjffhKLlAW23e0Z7IoEQ37tEPf_SE"
CHAT_ID = '955629733'
message_count = 0  # Counter for the number of detected infected plants
last_detection_time = 0  # Track last detection time
detection_cooldown = 5  # Cooldown between detections (seconds)

# Function to send message to Telegram
def send_telegram_message(count, image_path=None):
    try:
        message = f"I found {count} infected plant{'s' if count > 1 else ''}."
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={message}"
        r = requests.get(url, timeout=5)
        print(r.json())

        if image_path:
            with open(image_path, 'rb') as photo:  # Use context manager
                files = {'photo': photo}
                url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={CHAT_ID}"
                r = requests.post(url, files=files, timeout=5)
                print(r.json())
    except Exception as e:
        print(f"Telegram error: {e}")

# Initialize Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Load the YOLO model
model = YOLO("best.pt")

# Fixed FPS control
TARGET_FPS = 10
frame_delay = 1.0 / TARGET_FPS

try:
    while True:
        start_time = time.time()
        
        # Capture a frame from the camera
        frame = picam2.capture_array()

        # Perform detection using YOLO - DISABLE YOLO'S NATIVE PREVIEW
        results = model(frame, conf=0.3, show=False, verbose=False)  # Critical change
        result = results[0]

        # Get detection results
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        confidences = np.array(result.boxes.conf.cpu(), dtype="float")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")

        # Track detection time to avoid duplicates
        current_time = time.time()
        can_detect = (current_time - last_detection_time) > detection_cooldown

        for bbox, confi, cls in zip(bboxes, confidences, classes):
            (x, y, x2, y2) = bbox
            class_id = int(cls)
            object_name = model.names[class_id]
            
            # Draw bounding boxes and labels on the frame
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{object_name} {confi:.2f}", 
                        (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

            # Detection logic with cooldown
            if object_name.lower() == 'black-citron-aphid' and can_detect:
                last_detection_time = current_time
                message_count += 1
                
                # Save the cropped image
                infected_plant_image_path = f"infected_{int(time.time())}.jpg"
                cv2.imwrite(infected_plant_image_path, frame[y:y2, x:x2])
                
                # Send notification
                send_telegram_message(message_count, infected_plant_image_path)

        # Show detection count on screen
        cv2.putText(frame, f"Detected: {message_count}", (10, 30), 
                   cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        
        # Show the processed frame
        cv2.imshow("Plant Disease Detection", frame)

        # Frame rate control - critical for stable window
        processing_time = time.time() - start_time
        wait_time = max(1, int((frame_delay - processing_time) * 1000))
        
        # Check for exit command
        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q'):
            break

except Exception as e:
    print(f"Error: {str(e)}")
finally:
    # Release the camera and close any OpenCV windows
    picam2.stop()
    cv2.destroyAllWindows()
