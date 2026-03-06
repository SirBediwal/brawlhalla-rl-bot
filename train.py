from ultralytics import YOLO
import torch_directml

def start_training():
    # Wir nutzen das DirectML Device für AMD
    device = torch_directml.device()
    print(f"AMD Power aktiviert! Training läuft auf: {device}")

    # 1. Modell laden
    model = YOLO("yolo11n.pt") 

    # 2. Training starten
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=device, # Hier wird jetzt die 7900 XTX genutzt!
        name="brawli_amd_v1"
    )

if __name__ == "__main__":
    start_training()
