import mss
import cv2
import numpy as np
import keyboard
import os
import time
from datetime import datetime

# Einstellungen
SAVE_FOLDER = "data/raw_screenshots"
INTERVAL = 0.5  # Alle 0.5 Sekunden ein Bild (2 FPS reichen fuer den Anfang)
KEY_TOGGLE = "f9"  # Mit F9 starten/stoppen wir die Aufnahme

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

def main():
    sct = mss.mss()
    # Monitor 1 - passt die Auflösung ggf. an (1920x1080)
    monitor = sct.monitors[1]
    
    recording = False
    print(f"--- BRAWLHALLA DATA COLLECTOR ---")
    print(f"Druecke [{KEY_TOGGLE}], um die Aufnahme zu starten/stoppen.")
    print("Druecke [ESC], um das Programm zu beenden.")

    last_time = time.time()

    while True:
        if keyboard.is_pressed(KEY_TOGGLE):
            recording = not recording
            state = "GESTARTET" if recording else "GESTOPPT"
            print(f"Aufnahme {state}")
            time.sleep(0.5) # Debounce

        if recording:
            if time.time() - last_time > INTERVAL:
                # Screenshot machen
                img = np.array(sct.grab(monitor))
                # Von BGRA zu BGR konvertieren (OpenCV Standard)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # Dateiname mit Zeitstempel
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(SAVE_FOLDER, f"shot_{timestamp}.jpg")
                
                # Speichern (Qualität 90 reicht aus und spart Platz)
                cv2.imwrite(filename, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                last_time = time.time()

        if keyboard.is_pressed("esc"):
            print("Beende...")
            break

if __name__ == "__main__":
    main()
