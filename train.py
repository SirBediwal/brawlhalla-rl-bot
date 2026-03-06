from ultralytics import YOLO

def start_training():
    print("--- STARTE TRAINING AUF DER CPU ---")
    print("AMD-Karte wird ignoriert, um den DirectML-Fehler zu umgehen.")
    
    # Modell laden
    model = YOLO("yolo11n.pt") 

    # Training starten - ACHTUNG: device="cpu" erzwingt den Prozessor!
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,           
        device="cpu",      # <--- Hier darf NICHT 'device' oder 'privateuseone' stehen!
        name="brawli_cpu",
        plots=True
    )

if __name__ == "__main__":
    start_training()
