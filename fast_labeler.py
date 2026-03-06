import cv2
import os
import shutil
from ultralytics import YOLO

def run_fast_labeler():
    # 1. WICHTIG: Pfad zu eurem NEUEN 2-Klassen-Modell!
    # Checkt kurz, ob der Ordner wirklich brawli_2classes_v1 heißt.
    model_path = "runs/detect/brawli_2classes_v1/weights/best.pt"
    input_folder = "data/raw_screenshots" 
    
    out_images = "dataset/train/images"
    out_labels = "dataset/train/labels"

    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    # 2. Modell laden
    print(f"Lade KI-Modell von: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception:
        print("Fehler: Modell nicht gefunden! Stimmt der Pfad in Zeile 9?")
        return

    print("\n--- FAST LABELER (2 KLASSEN) GESTARTET ---")
    print("Prüfe, ob die KI 'me' und 'enemy' richtig erkannt hat!")
    print(" [ Y ] = PERFEKT! Beide richtig erkannt -> Speichern.")
    print(" [ N ] = FALSCH (z.B. vertauscht). -> Überspringen.")
    print(" [ Q ] = BEENDEN.")
    print("------------------------------------------\n")

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(input_folder, filename)
        
        # KI raten lassen
        results = model.predict(img_path, conf=0.4, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            continue

        # Bild mit gezeichneten Boxen UND NAMEN anzeigen
        annotated_frame = results[0].plot()
        
        cv2.imshow("Fast Labeler (Y=Ja, N=Nein, Q=Ende)", cv2.resize(annotated_frame, (960, 540)))

        key = cv2.waitKey(0) & 0xFF

        if key == ord('y'):
            # BILD WAR GUT -> Speichern!
            shutil.copy(img_path, os.path.join(out_images, filename))
            
            txt_filename = filename.replace(".jpg", ".txt").replace(".png", ".txt")
            with open(os.path.join(out_labels, txt_filename), "w") as f:
                for i in range(len(boxes)):
                    # HIER IST DIE MAGIE: Wir nehmen jetzt die echte Klasse der KI (0 oder 1)
                    cls = int(boxes.cls[i].item())
                    x, y, w, h = boxes.xywhn[i].tolist()
                    
                    # Speichert dynamisch 0 oder 1 in die Textdatei
                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n") 
            
            print(f"[+] GESPEICHERT: {filename}")
            os.remove(img_path) 

        elif key == ord('q'):
            print("Beendet!")
            break
        else:
            print(f"[-] ÜBERSPRUNGEN: {filename}")
            os.remove(img_path)

    cv2.destroyAllWindows()
    print("\nFertig! Deine perfekten Daten liegen im dataset/train Ordner.")

if __name__ == "__main__":
    run_fast_labeler()
