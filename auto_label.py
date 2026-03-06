import os
import shutil
from ultralytics import YOLO

def start_auto_labeling():
    # 1. Lade euer trainiertes Modell
    # WICHTIG: Prüft, in welchem Ordner euer best.pt genau liegt!
    model_path = "runs/detect/brawli_cpu/weights/best.pt"
    model = YOLO(model_path)

    # 2. Ordner-Pfade einstellen
    # Hier sollten eure neuen, noch ungelabelten Screenshots liegen (z.B. vom collect_data.py)
    input_folder = "data/raw_screenshots" 
    
    # Hier speichert das Skript die fertigen Bilder und Labels
    output_images = "dataset/auto_labeled/images"
    output_labels = "dataset/auto_labeled/labels"

    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)

    print(f"Starte Auto-Labeling für Bilder in: {input_folder}")

    # 3. Alle Bilder im Ordner durchgehen
    for filename in os.listdir(input_folder):
        if not filename.endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(input_folder, filename)

        # Die KI raten lassen (conf=0.4 bedeutet: Er soll auch Boxen zeichnen, 
        # wenn er sich nur zu 40% sicher ist. Korrigieren könnt ihr das später ja eh!)
        results = model.predict(img_path, conf=0.4, verbose=False)
        boxes = results[0].boxes

        # Nur Bilder speichern, auf denen er auch was gefunden hat
        if len(boxes) > 0:
            # Bild in den neuen Ordner kopieren
            shutil.copy(img_path, os.path.join(output_images, filename))

            # Die .txt Datei mit den Koordinaten erstellen
            txt_filename = filename.replace(".jpg", ".txt").replace(".png", ".txt")
            txt_path = os.path.join(output_labels, txt_filename)

            with open(txt_path, "w") as f:
                for i in range(len(boxes)):
                    # Klasse und Koordinaten (normiert für YOLO) auslesen
                    cls = int(boxes.cls[i].item())
                    x, y, w, h = boxes.xywhn[i].tolist()
                    
                    # In Datei schreiben
                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                    
    print(f"Fertig! Die vorbereiteten Daten liegen in dataset/auto_labeled/")

if __name__ == "__main__":
    start_auto_labeling()
