from ultralytics import YOLO
import torch

def start_training():
    # Prüfen, ob eine NVIDIA GPU verfügbar ist (macht es 10x schneller)
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Training läuft auf: {device}")

    # 1. Ein vortrainiertes Modell laden (YOLOv11 Nano - perfekt für Games)
    model = YOLO("yolo11n.pt") 

    # 2. Das Training starten
    model.train(
        data="data.yaml",     # Unsere Landkarte von eben
        epochs=50,            # Wie oft die KI die Bilder "studiert"
        imgsz=640,            # Bildgröße für das Training
        batch=16,             # Wie viele Bilder gleichzeitig verarbeitet werden
        device=device,        # GPU oder CPU
        name="brawli_v1"      # Name des Versuchs
    )

if __name__ == "__main__":
    start_training()
