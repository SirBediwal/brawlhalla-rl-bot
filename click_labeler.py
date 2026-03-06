import cv2
import os
import shutil

# Pfade konfigurieren
input_folder = "data/raw_screenshots"
out_images = "dataset/train/images"
out_labels = "dataset/train/labels"

os.makedirs(out_images, exist_ok=True)
os.makedirs(out_labels, exist_ok=True)

# Globale Variablen für das Zeichnen
boxes = []
drawing = False
ix, iy = -1, -1
current_class = 0 
img_display = None
img_clean = None

def draw_box(event, x, y, flags, param):
    global ix, iy, drawing, img_display, img_clean, boxes, current_class

    # LINKSKLICK: Klasse 0 (ME - Grün)
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        current_class = 0 
        
    # RECHTSKLICK: Klasse 1 (ENEMY - Rot)
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        current_class = 1 
        
    # MAUS BEWEGEN (Vorschau der Box zeichnen)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_display = img_clean.copy()
            color = (0, 255, 0) if current_class == 0 else (0, 0, 255)
            cv2.rectangle(img_display, (ix, iy), (x, y), color, 2)
            
    # MAUSTASTE LOSLASSEN (Box speichern)
    elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
        if drawing:
            drawing = False
            color = (0, 255, 0) if current_class == 0 else (0, 0, 255)
            cv2.rectangle(img_clean, (ix, iy), (x, y), color, 2)
            img_display = img_clean.copy()
            
            # YOLO Format berechnen (Werte zwischen 0 und 1)
            h_img, w_img = img_clean.shape[:2]
            x_min, x_max = min(ix, x), max(ix, x)
            y_min, y_max = min(iy, y), max(iy, y)
            
            w = x_max - x_min
            h = y_max - y_min
            
            # Verhindern, dass aus Versehen leere Klicks als Box gespeichert werden
            if w > 5 and h > 5:
                x_center = (x_min + w / 2) / w_img
                y_center = (y_min + h / 2) / h_img
                w_norm = w / w_img
                h_norm = h / h_img
                boxes.append((current_class, x_center, y_center, w_norm, h_norm))

def run_click_labeler():
    global img_display, img_clean, boxes
    
    cv2.namedWindow("Click Labeler")
    cv2.setMouseCallback("Click Labeler", draw_box)
    
    print("\n--- KLICK-LABELER GESTARTET ---")
    print("Mausklick + Ziehen, um eine Box zu malen:")
    print(" [ LINKSKLICK ]  = ME    (Gruene Box)")
    print(" [ RECHTSKLICK ] = ENEMY (Rote Box)")
    print("-----------------------------------")
    print(" [ LEERTASTE ] = Speichern & Naechstes Bild")
    print(" [ C ] = Boxen loeschen (Clear)")
    print(" [ S ] = Bild ueberspringen (Skip)")
    print(" [ Q ] = Beenden\n")

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(input_folder, filename)
        img_clean = cv2.imread(img_path)
        if img_clean is None: continue
        
        img_display = img_clean.copy()
        boxes = []
        
        while True:
            cv2.imshow("Click Labeler", img_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '): # LEERTASTE -> Speichern
                if len(boxes) > 0:
                    shutil.copy(img_path, os.path.join(out_images, filename))
                    txt_name = filename.rsplit('.', 1)[0] + '.txt'
                    with open(os.path.join(out_labels, txt_name), "w") as f:
                        for b in boxes:
                            # Schreibt genau das richtige YOLO-Format in die Datei
                            f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
                    print(f"[+] Gespeichert: {filename} mit {len(boxes)} Box(en)")
                    os.remove(img_path) # Original loeschen
                else:
                    print("[-] Keine Boxen gezeichnet. Bild wird uebersprungen.")
                    os.remove(img_path)
                break
                
            elif key == ord('c'): # CLEAR -> Neu zeichnen
                img_clean = cv2.imread(img_path)
                img_display = img_clean.copy()
                boxes = []
                print("Boxen geloescht. Zeichne neu!")
                
            elif key == ord('s'): # SKIP -> Verwerfen
                print(f"[-] Uebersprungen: {filename}")
                os.remove(img_path)
                break
                
            elif key == ord('q'): # QUIT -> Beenden
                print("Beendet!")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("Alle Bilder im Ordner bearbeitet!")

if __name__ == "__main__":
    run_click_labeler()
