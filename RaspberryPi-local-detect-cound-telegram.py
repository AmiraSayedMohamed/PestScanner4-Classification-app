from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np
import cv2
import requests
import time

# Telegram bot parameters
TOKEN = "7290187905:AAHp7vnjffhKLlAW23e0Z7IoEQ37tEPf_SE"
CHAT_ID = '955629733'
message_count = 0
last_detection_time = 0
detection_cooldown = 5

def send_telegram_message(count, image_path=None):
    try:
        message = f"I found {count} infected plant{'s' if count > 1 else ''}."
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={message}"
        r = requests.get(url, timeout=5)
        print(r.json())

        if image_path:
            with open(image_path, 'rb') as photo:
                files = {'photo': photo}
                url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={CHAT_ID}"
                r = requests.post(url, files=files, timeout=5)
                print(r.json())
    except Exception as e:
        print(f"Telegram error: {e}")

# Initialize Picamera2 with proper configuration
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}  # Explicitly request RGB format
)
picam2.configure(config)
picam2.start()

model = YOLO("best.pt")
TARGET_FPS = 10
frame_delay = 1.0 / TARGET_FPS

try:
    while True:
        start_time = time.time()
        
        frame = picam2.capture_array()
        
        # Ensure 3-channel input for YOLO (convert if necessary)
        if frame.shape[2] == 4:  # If RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif frame.shape[2] == 1:  # If grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        
        # Perform detection
        results = model(frame, conf=0.3, show=False, verbose=False)
        result = results[0]

        # Process results
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        confidences = np.array(result.boxes.conf.cpu(), dtype="float")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")

        current_time = time.time()
        can_detect = (current_time - last_detection_time) > detection_cooldown

        for bbox, confi, cls in zip(bboxes, confidences, classes):
            (x, y, x2, y2) = bbox
            class_id = int(cls)
            object_name = model.names[class_id]
            
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{object_name} {confi:.2f}", 
                       (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

            if object_name.lower() == 'black-citron-aphid' and can_detect:
                last_detection_time = current_time
                message_count += 1
                infected_plant_image_path = f"infected_{int(time.time())}.jpg"
                cv2.imwrite(infected_plant_image_path, frame[y:y2, x:x2])
                send_telegram_message(message_count, infected_plant_image_path)

        cv2.putText(frame, f"Detected: {message_count}", (10, 30), 
                   cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv2.imshow("Plant Disease Detection", frame)

        processing_time = time.time() - start_time
        wait_time = max(1, int((frame_delay - processing_time) * 1000))
        
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {str(e)}")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
