from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np
import cv2
import requests
from queue import Queue
import threading
import time
import os

# Telegram bot parameters
TOKEN = "7290187905:AAHp7vnjffhKLlAW23e0Z7IoEQ37tEPf_SE"
CHAT_ID = '955629733'
message_count = 0  # Counter for the number of detected infected plants

# Function to send message to Telegram
def send_telegram_message(count, image_path=None):
    message = f"I found {count} infected plant{'s' if count > 1 else ''}."
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={message}"
    r = requests.get(url)
    print(r.json())

    if image_path:
        files = {'photo': open(image_path, 'rb')}
        url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={CHAT_ID}"
        r = requests.post(url, files=files)
        print(r.json())

# Telegram worker function to handle messages in a separate thread
def telegram_worker():
    while True:
        task = telegram_queue.get()
        if task is None:
            break
        count, image_path = task
        send_telegram_message(count, image_path)
        if image_path:
            try:
                os.remove(image_path)  # Clean up the image file after sending
            except Exception as e:
                print(f"Error deleting image: {e}")

# Initialize Picamera2 with lower resolution and BGR format for OpenCV
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240), "format": "BGR888"}))
picam2.start()

# Load the YOLO model
model = YOLO("best.pt")

# Create Telegram queue and start worker thread
telegram_queue = Queue()
telegram_thread = threading.Thread(target=telegram_worker, daemon=True)
telegram_thread.start()

# Create and position the OpenCV window once
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.moveWindow("Camera", 100, 100)  # Fix window position

try:
    while True:
        # Capture a frame from the camera (in BGR format)
        frame = picam2.capture_array()

        # Convert frame to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform detection using YOLO without its own window
        results = model(frame_rgb, show=False, conf=0.3)
        result = results[0]

        # Get bounding boxes, confidence, and class names
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        confidences = np.array(result.boxes.conf.cpu(), dtype="float")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")

        for bbox, confi, cls in zip(bboxes, confidences, classes):
            (x, y, x2, y2) = bbox
            class_id = int(cls)
            object_name = model.names[class_id]
            if object_name.lower() == 'black-citron-aphid':
                message_count += 1
                # Save image with timestamp to ensure unique filenames
                infected_plant_image_path = f"infected_{message_count}_{int(time.time())}.jpg"
                cv2.imwrite(infected_plant_image_path, frame[y:y2, x:x2])
                # Send to Telegram via queue
                telegram_queue.put((message_count, infected_plant_image_path))

            # Draw bounding boxes and labels on the frame
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{object_name} {confi:.2f}", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        # Show the processed frame in the fixed window
        cv2.imshow("Camera", frame)

        # Exit on 'q' key press, with a slight delay for stability
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupt received. Closing...")
finally:
    # Clean up
    picam2.stop()
    cv2.destroyAllWindows()
    telegram_queue.put(None)  # Signal the worker thread to stop
