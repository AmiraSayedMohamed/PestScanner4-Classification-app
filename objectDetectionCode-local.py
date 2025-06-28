# this code is an object detection code  and sent the detected image to telegram bot and count the infected  plsny

import cv2
import numpy as np
from ultralytics import YOLO
import requests
import threading
from queue import Queue
import time
import os
from flask import Flask, Response, render_template_string

# Initialize Flask app
app = Flask(__name__)

# Telegram bot parameters
TOKEN = "7290187905:AAHp7vnjffhKLlAW23e0Z7IoEQ37tEPf_SE"
CHAT_ID = '955629733'
message_count = 0

# Communication queues
frame_queue = Queue(maxsize=1)
result_queue = Queue(maxsize=1)
telegram_queue = Queue()
web_queue = Queue(maxsize=1)  # For frames to be displayed in web app

# Global variables for camera control
camera_active = True

# Function to send message to Telegram
def telegram_worker():
    while True:
        task = telegram_queue.get()
        if task is None:  # Exit signal
            break
            
        count, image_path = task
        message = f"I found {count} infected plant{'s' if count > 1 else ''}."
        
        try:
            if image_path:
                with open(image_path, 'rb') as photo:
                    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
                    r = requests.post(url, 
                                    files={'photo': photo},
                                    data={'chat_id': CHAT_ID, 'caption': message},
                                    timeout=10)
                os.remove(image_path)
            else:
                url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
                r = requests.post(url,
                                data={'chat_id': CHAT_ID, 'text': message},
                                timeout=5)
            print("Telegram response:", r.status_code)
        except Exception as e:
            print(f"Telegram error: {str(e)}")

# Worker function for model inference
def model_worker():
    model = YOLO("best.pt")
    while True:
        frame = frame_queue.get()
        if frame is None:  # Exit signal
            break
            
        results = model(frame, conf=0.3, verbose=False)
        result_queue.put(results[0])

# Main processing function
def camera_processing():
    global message_count, camera_active
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    last_detection_time = 0
    detection_cooldown = 5
    frame_skip = 2
    frame_counter = 0

    try:
        while camera_active:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Camera error")
                break

            frame_counter += 1
            
            if frame_counter % frame_skip != 0:
                continue

            if frame_queue.empty():
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_queue.put(frame_rgb)

            display_frame = frame.copy()
            if not result_queue.empty():
                result = result_queue.get()
                bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
                confidences = np.array(result.boxes.conf.cpu(), dtype="float")
                classes = np.array(result.boxes.cls.cpu(), dtype="int")

                current_time = time.time()
                can_detect = (current_time - last_detection_time) > detection_cooldown

                for bbox, confi, cls in zip(bboxes, confidences, classes):
                    (x, y, x2, y2) = bbox
                    class_id = int(cls)
                    object_name = result.names[class_id]
                    
                    cv2.rectangle(display_frame, (x, y), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(display_frame, f"{object_name} {confi:.2f}",
                              (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

                    if object_name.lower() == 'black-citrus-aphid' and can_detect:
                        last_detection_time = current_time
                        message_count += 1
                        
                        detection_img = frame[y:y2, x:x2]
                        temp_path = f"temp_infected_{int(time.time())}.jpg"
                        cv2.imwrite(temp_path, detection_img)
                        
                        telegram_queue.put((message_count, temp_path))

            # Put frame in web queue if empty
            if web_queue.empty():
                ret, jpeg = cv2.imencode('.jpg', display_frame)
                web_queue.put(jpeg.tobytes())

            proc_time = time.time() - start_time
            if proc_time > 0.1:
                frame_skip = min(5, frame_skip + 1)
            elif frame_skip > 1:
                frame_skip = max(1, frame_skip - 1)

    except Exception as e:
        print(f"Camera processing error: {str(e)}")
    finally:
        cap.release()
        frame_queue.put(None)
        telegram_queue.put(None)

# Flask routes
@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pest Detection System</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 20px; }
            h1 { color: #2c3e50; }
            .container { max-width: 800px; margin: 0 auto; }
            .stats { background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0; }
            img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Pest Detection System</h1>
            <div class="stats">
                <p>Detected pests: {{ count }}</p>
            </div>
            <img src="{{ url_for('video_feed') }}" width="640" height="480">
        </div>
    </body>
    </html>
    """, count=message_count)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if not web_queue.empty():
                frame = web_queue.get()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.01)
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_active
    camera_active = not camera_active
    return {'status': 'success', 'camera_active': camera_active}

def cleanup():
    global camera_active
    camera_active = False
    for f in os.listdir():
        if f.startswith('temp_infected_') and f.endswith('.jpg'):
            try:
                os.remove(f)
            except:
                pass

if __name__ == '__main__':
    # Start worker threads
    model_thread = threading.Thread(target=model_worker, daemon=True)
    telegram_thread = threading.Thread(target=telegram_worker, daemon=True)
    camera_thread = threading.Thread(target=camera_processing, daemon=True)
    
    model_thread.start()
    telegram_thread.start()
    camera_thread.start()
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        cleanup()
