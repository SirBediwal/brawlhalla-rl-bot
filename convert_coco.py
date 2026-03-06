import json
import os

def convert_coco_to_yolo(json_path, output_dir):
    if not os.path.exists(json_path):
        print(f"Fehler: Datei {json_path} nicht gefunden!")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Mapping von Image-ID zu Info
    images = {img['id']: img for img in data['images']}
    
    for ann in data['annotations']:
        img_id = ann['image_id']
        img_info = images[img_id]
        
        # Hier erzwingen wir Zahlenwerte (float)
        img_w = float(img_info['width'])
        img_h = float(img_info['height'])
        
        file_name = img_info['file_name'].replace('.jpg', '.txt')
        
        # Bbox Werte sicher in Floats umwandeln
        x, y, w, h = map(float, ann['bbox'])
        
        # Berechnung der relativen YOLO-Koordinaten
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        
        class_id = int(ann['category_id'])
        
        # In .txt Datei schreiben
        with open(os.path.join(output_dir, file_name), 'a') as out_f:
            line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
            out_f.write(line)

    print(f"Erfolg! Labels wurden in {output_dir} gespeichert.")

if __name__ == "__main__":
    # Achtet auf die Pfade - passt sie an, wenn euer Ordner 'datasets' (mit s) heißt!
    json_file = "dataset/train/_annotations.coco.json"
    label_folder = "dataset/train/labels"
    convert_coco_to_yolo(json_file, label_folder)
