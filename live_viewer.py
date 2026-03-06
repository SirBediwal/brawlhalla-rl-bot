from ultralytics import YOLO
import cv2
import bettercam

def run_live_viewer():
    # 1. WICHTIG: Pfad zu eurem AKTUELLSTEN Modell anpassen!
    # Schaut in den Ordner runs/detect/..., wie euer letzter Run hieß
    model_path = "runs/detect/brawli_cpu3/weights/best.pt" 
    
    print(f"Lade KI-Modell von: {model_path} ...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Fehler: Konnte best.pt nicht finden! Stimmt der Pfad? ({e})")
        return

    # 2. Bildschirmaufnahme starten (mit BetterCam für max FPS)
    camera = bettercam.create()
    camera.start(target_fps=60)

    print("\n--- BRAWLI LIVE VISION AKTIVIERT ---")
    print("Geh ins Spiel! Druecke 'q' im Kamera-Fenster, um es zu beenden.")
    print("------------------------------------\n")

    while True:
        # Aktuelles Bild vom Bildschirm holen
        frame = camera.get_latest_frame()
        
        if frame is None:
            continue

        # WICHTIG: Farben von RGB zu BGR umwandeln (Euer "Schlumpf-Filter" Fix!)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 3. Die KI sucht nach den Charakteren
        # conf=0.45 bedeutet: Er muss sich zu 45% sicher sein, um eine Box zu malen
        results = model.predict(frame_bgr, conf=0.45, verbose=False)

        # 4. Boxen (und Namen) automatisch auf das Bild zeichnen lassen
        annotated_frame = results[0].plot()

        # 5. Bild in einem Overlay-Fenster anzeigen (Skaliert auf halbe Größe für Übersicht)
        cv2.imshow("Brawli Live Vision", cv2.resize(annotated_frame, (960, 540)))

        # 6. Mit 'q' beenden
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Aufräumen
    camera.stop()
    cv2.destroyAllWindows()
    print("Live Vision beendet.")

if __name__ == "__main__":
    run_live_viewer()
