import cv2
import numpy as np
from ultralytics import YOLO
import requests
import threading
from queue import Queue
import time
import os
from picamera2 import Picamera2

# Telegram bot parameters
TOKEN = "7290187905:AAHp7vnjffhKLlAW23e0Z7IoEQ37tEPf_SE"
CHAT_ID = '955629733'
message_count = 0  # Counter for the number of detected infected plants

# Communication queues
frame_queue = Queue(maxsize=1)
result_queue = Queue(maxsize=2)  # Increased to buffer results

# Function to send message to Telegram (runs in a separate thread)
def telegram_worker():
    while True:
        task = telegram_queue.get()
        if task is None:  # Exit signal
            break
            
        count, image_path = task
        message = f"I found {count} infected plant{'s' if count > 1 else ''}."
        
        try:
            # Send text message
            url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={message}"
            r = requests.get(url, timeout=10)
            print(f"Telegram text response: {r.status_code}, {r.json()}")

            # Send image if available
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as photo:
                    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={CHAT_ID}"
                    r = requests.post(url, files={'photo': photo}, timeout=10)
                    print(f"Telegram photo response: {r.status_code}, {r.json()}")
                # Delete the temporary image after sending
                try:
                    os.remove(image_path)
                    print(f"Deleted image: {image_path}")
                except Exception as e:
                    print(f"Error deleting image {image_path}: {str(e)}")
            else:
                print(f"Image not found: {image_path}")
        except Exception as e:
            print(f"Telegram error: {str(e)}")

# Worker function for model inference
def model_worker():
    model = YOLO("best.pt")
    while True:
        task = frame_queue.get()
        if task is None:  # Exit signal
            break
            
        frame_id, frame_rgb, frame_bgr = task
        try:
            # Perform detection
            results = model(frame_rgb, conf=0.3, verbose=False)  # show=False for stability
            result_queue.put((frame_id, results[0], frame_bgr))  # Store frame_id, result, and BGR frame
        except Exception as e:
            print(f"Inference error: {str(e)}")
            result_queue.put((frame_id, None, None))  # Put None to avoid blocking

def main():
    global message_count
    
    # Start worker threads
    model_thread = threading.Thread(target=model_worker, daemon=True)
    telegram_thread = threading.Thread(target=telegram_worker, daemon=True)
    model_thread.start()
    telegram_thread.start()

    # Initialize Picamera2 with larger resolution
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (800, 600), "format": "BGR888"})
    picam2.configure(config)
    try:
        picam2.start()
    except Exception as e:
        print(f"Camera initialization error: {str(e)}")
        return

    # Create OpenCV window once, set fixed position, and resize to match resolution
    cv2.namedWindow("Pest Detection", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Pest Detection", 100, 100)  # Fixed position (x=100, y=100)
    cv2.resizeWindow("Pest Detection", 800, 600)  # Match camera resolution

    # Performance tracking
    last_detection_time = 0
    detection_cooldown = 5  # Minimum seconds between Telegram notifications
    frame_skip = 4  # Frame skip for lighter processing
    frame_counter = 0
    frame_id = 0  # Unique ID for each frame

    try:
        while True:
            start_time = time.time()
            
            # Capture a frame from the camera
            try:
                frame = picam2.capture_array("main")
                if frame is None:
                    print("Failed to capture frame")
                    continue
            except Exception as e:
                print(f"Frame capture error: {str(e)}")
                continue

            # Frame is already in BGR format (BGR888)
            frame_bgr = frame
            frame_counter += 1
            frame_id += 1

            # Skip processing frames if needed to reduce load
            if frame_counter % frame_skip != 0:
                cv2.putText(frame_bgr, f"Detected pests: {message_count}", (10, 30),
                           cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                try:
                    cv2.imshow("Pest Detection", frame_bgr)
                except Exception as e:
                    print(f"Display error: {str(e)}")
                if cv2.waitKey(10) & 0xFF == ord('q'):  # 10ms delay for stability
                    break
                continue

            # Process frame if queue is ready
            if frame_queue.empty():
                # Convert to RGB for YOLO
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_queue.put((frame_id, frame_rgb, frame_bgr))

            # Check for results
            display_frame = frame_bgr.copy()
            if not result_queue.empty():
                result_frame_id, result, result_frame_bgr = result_queue.get()
                if result is None or result_frame_bgr is None:
                    print(f"No valid result for frame_id {result_frame_id}")
                    continue

                # Use the frame that matches the detection result
                display_frame = result_frame_bgr.copy()
                print(f"Processing results for frame_id {result_frame_id}")

                bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
                confidences = np.array(result.boxes.conf.cpu(), dtype="float")
                classes = np.array(result.boxes.cls.cpu(), dtype="int")

                current_time = time.time()
                can_send_telegram = (current_time - last_detection_time) > detection_cooldown

                for bbox, confi, cls in zip(bboxes, confidences, classes):
                    (x, y, x2, y2) = bbox
                    class_id = int(cls)
                    object_name = result.names[class_id]
                    
                    # Draw bounding boxes and labels for ALL detections
                    cv2.rectangle(display_frame, (x, y), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(display_frame, f"{object_name} {confi:.2f}",
                               (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                    print(f"Drawing bounding box for {object_name} at ({x}, {y}, {x2}, {y2}) with conf {confi:.2f}")

                    # Handle Telegram notification for infected plants
                    if object_name.lower() == 'black-citrus-aphid' and can_send_telegram:
                        last_detection_time = current_time
                        message_count += 1

                        # Save the image of the infected plant
                        temp_path = f"infected_{message_count}_{int(time.time())}.jpg"
                        try:
                            cv2.imwrite(temp_path, result_frame_bgr[y:y2, x:x2])
                            print(f"Saved detection image: {temp_path}")
                            telegram_queue.put((message_count, temp_path))
                        except Exception as e:
                            print(f"Image save error: {str(e)}")

            # Display count on frame
            cv2.putText(display_frame, f"Detected pests: {message_count}",
                       (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

            # Show the processed frame
            try:
                cv2.imshow("Pest Detection", display_frame)
            except Exception as e:
                print(f"Display error: {str(e)}")

            # Adjust frame skipping based on processing time
            proc_time = time.time() - start_time
            if proc_time > 0.1:  # If processing is slow
                frame_skip = min(8, frame_skip + 1)
            elif frame_skip > 4:  # If we can afford to process more
                frame_skip = max(4, frame_skip - 1)

            # Exit on 'q' key with stable delay
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupt received. Closing...")
    except Exception as e:
        print(f"Main loop error: {str(e)}")
    finally:
        # Cleanup
        frame_queue.put(None)
        result_queue.put((None, None, None))
        telegram_queue.put(None)
        picam2.stop()
        cv2.destroyWindow("Pest Detection")
        cv2.destroyAllWindows()
        # Remove any remaining temp files
        for f in os.listdir():
            if f.startswith('infected_') and f.endswith('.jpg'):
                try:
                    os.remove(f)
                except:
                    pass

if __name__ == "__main__":
    main()
