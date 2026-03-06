import bettercam
import cv2
import numpy as np
import keyboard
import os
import time
import threading
from queue import Queue
from datetime import datetime

# Einstellungen
SAVE_FOLDER = "data/raw_screenshots"
INTERVAL = 0.5 
TARGET_SIZE = (640, 640) # KI-Modelle brauchen oft nur 640px, spart massiv Speicher/Zeit!

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Warteschlange fuer Bilder
image_queue = Queue()

def save_worker():
    """Dieser Thread kuesst die Performance, indem er im Hintergrund speichert."""
    while True:
        img, filename = image_queue.get()
        if img is None: break
        # Resize direkt beim Speichern spart I/O Last
        img_resized = cv2.resize(img, TARGET_SIZE)
        cv2.imwrite(filename, img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        image_queue.task_done()

def main():
    # BetterCam initialisieren (extrem schnell auf Windows)
    camera = bettercam.create()
    
    # Speicher-Thread starten
    worker = threading.Thread(target=save_worker, daemon=True)
    worker.start()

    recording = False
    print("--- BRAWLI OPTIMIZED COLLECTOR ---")
    print("F9: Start/Stop | ESC: Beenden")

    last_time = time.time()

    while True:
        if keyboard.is_pressed("f9"):
            recording = not recording
            print(f"Aufnahme {'AKTIVIERT' if recording else 'PAUSIERT'}")
            time.sleep(0.5)

        if recording and (time.time() - last_time > INTERVAL):
            # Frame direkt von der GPU holen
            frame = camera.grab()
            
            if frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(SAVE_FOLDER, f"shot_{timestamp}.jpg")
                
                # Bild in die Queue werfen statt sofort zu speichern
                image_queue.put((frame, filename))
                last_time = time.time()

        if keyboard.is_pressed("esc"):
            image_queue.put((None, None)) # Worker stoppen
            break

if __name__ == "__main__":
    main()
