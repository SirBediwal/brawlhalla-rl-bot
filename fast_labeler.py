import cv2
import os
import shutil
from ultralytics import YOLO

def run_fast_labeler():
    # 1. Pfade konfigurieren
    model_path = "yolo11n.pt"
    input_folder = "data/raw_screenshots" 
    
    # Hier speichern wir die GUTE Beute direkt ab, bereit fürs nächste Training!
    out_images = "dataset/train/images"
    out_labels = "dataset/train/labels"

    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    # 2. Modell laden
    print(f"Lade KI-Modell von: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception:
        print("Fehler: Modell nicht gefunden!")
        return

    print("\n--- FAST LABELER GESTARTET ---")
    print("Steuerung im Bild-Fenster:")
    print(" [ Y ] = PERFEKT! Speichern.")
    print(" [ N ] = FALSCH/UNGENAU. Überspringen.")
    print(" [ Q ] = BEENDEN.")
    print("------------------------------\n")

    # Alle Bilder im Ordner durchgehen
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(input_folder, filename)
        
        # KI raten lassen
        results = model.predict(img_path, conf=0.4, verbose=False)
        boxes = results[0].boxes

        # Wenn er NICHTS findet, überspringen wir es direkt automatisch
        if len(boxes) == 0:
            continue

        # Bild mit gezeichneten Boxen anzeigen
        annotated_frame = results[0].plot()
        
        # Fenster etwas kleiner machen, damit es gut auf den Screen passt
        cv2.imshow("Fast Labeler (Y=Ja, N=Nein, Q=Ende)", cv2.resize(annotated_frame, (960, 540)))

        # Warten auf Tastendruck
        key = cv2.waitKey(0) & 0xFF

        if key == ord('y'):
            # BILD WAR GUT -> Speichern!
            # 1. Bild rüberkopieren
            shutil.copy(img_path, os.path.join(out_images, filename))
            
            # 2. YOLO Textdatei erstellen
            txt_filename = filename.replace(".jpg", ".txt").replace(".png", ".txt")
            with open(os.path.join(out_labels, txt_filename), "w") as f:
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i].item())
                    x, y, w, h = boxes.xywhn[i].tolist()
                    f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n") # Wir erzwingen Klasse 0
            
            print(f"[+] GESPEICHERT: {filename}")
            
            # (Optional) Original löschen, damit es nicht nochmal drankommt
            os.remove(img_path) 

        elif key == ord('q'):
            print("Beendet!")
            break
        else:
            # Bei 'N' oder jeder anderen Taste wird das Bild einfach übersprungen
            print(f"[-] ÜBERSPRUNGEN: {filename}")
            # (Optional) Original löschen, damit es nicht nochmal drankommt
            os.remove(img_path)

    cv2.destroyAllWindows()
    print("\nFertig! Deine neuen, perfekten Daten liegen im dataset/train Ordner.")

if __name__ == "__main__":
    run_fast_labeler()
