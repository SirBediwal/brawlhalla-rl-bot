import json
import os

def convert_coco_to_yolo(json_path, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Erstellt ein Mapping von Image-ID zu Dateiname und Groesse
    images = {img['id']: img for img in data['images']}
    
    for ann in data['annotations']:
        img_id = ann['image_id']
        img_info = images[img_id]
        
        img_w = img_info['width']
        img_h = img_info['height']
        file_name = img_info['file_name'].replace('.jpg', '.txt')
        
        # COCO: [x_min, y_min, width, height]
        # YOLO: [x_center, y_center, width, height] (normiert 0-1)
        x, y, w, h = ann['bbox']
        
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        
        # Klasse (In eurer JSON ist 0 = objects, 1 = player)
        class_id = ann['category_id']
        
        # In Datei schreiben (append mode)
        with open(os.path.join(output_dir, file_name), 'a') as out_f:
            line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
            out_f.write(line)

    print(f"Fertig! Labels wurden in {output_dir} gespeichert.")

if __name__ == "__main__":
    # Pfade anpassen, falls sie bei euch anders heissen
    json_file = "dataset/train/_annotations.coco.json"
    label_folder = "dataset/train/labels"
    convert_coco_to_yolo(json_file, label_folder)
