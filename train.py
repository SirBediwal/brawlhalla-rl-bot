from ultralytics import YOLO
import torch_directml

def start_training():
    # DirectML Device für die AMD-Karte
    device="cpu"
    print(f"AMD Power aktiviert! Training läuft auf: {device}")

    # Modell laden
    model = YOLO("yolo11n.pt") 

    # Training starten mit den AMD-Fixes
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        amp=False,          # <--- DAS IST DER ENTSCHEIDENDE FIX!
        name="brawli_amd_v1",
        plots=True
    )

if __name__ == "__main__":
    start_training()
