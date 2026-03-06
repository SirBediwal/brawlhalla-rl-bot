from ultralytics import YOLO
import cv2
import bettercam
import numpy as np

def run_bot_vision():
    # 1. Lade dein trainiertes Modell (Das ist euer Goldstück!)
    # WICHTIG: Prüft, ob der Pfad stimmt. Manchmal heißt der Ordner brawli_cpu2 oder brawli_cpu3
    model_path = "yolo11n.pt" 
    
    print(f"Lade KI-Modell von: {model_path} ...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print("Fehler: Konnte best.pt nicht finden! Stimmt der Pfad?")
        return

    # 2. Bildschirmaufnahme starten (BetterCam ist ultra schnell für Windows)
    camera = bettercam.create()
    camera.start(target_fps=60)

    print("--- BRAWLI VISION AKTIVIERT ---")
    print("Geh ins Spiel! Drücke 'q' im Kamera-Fenster, um es zu beenden.")

    while True:
        # 3. Aktuelles Bild vom Bildschirm holen
        frame = camera.get_latest_frame()
        
        if frame is None:
            continue

        # BetterCam liefert RGB, OpenCV arbeitet mit BGR (Farbräume anpassen)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 4. Die KI sucht nach dem Charakter 
        # conf=0.5 bedeutet: Zeichne die Box nur, wenn du dir zu 50% sicher bist
        results = model.predict(frame_bgr, conf=0.5, verbose=False)

        # 5. Boxen automatisch auf das Bild zeichnen lassen!
        # results[0].plot() ist ein genialer YOLO-Befehl, der das Bild fix und fertig zurückgibt
        annotated_frame = results[0].plot()

        # 6. Das Bild in einem kleinen Overlay-Fenster anzeigen
        # Wir machen das Fenster etwas kleiner, damit es nicht den ganzen echten Bildschirm verdeckt
        cv2.imshow("Brawli Auge (Live)", cv2.resize(annotated_frame, (960, 540)))

        # 7. Mit 'q' beenden
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Aufräumen, wenn wir fertig sind
    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_bot_vision()
