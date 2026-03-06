from ultralytics import YOLO

def start_training():
    print("--- STARTE TRAINING (RUNDE 2) AUF DER CPU ---")
    print("Die KI wird jetzt noch schlauer gemacht!")
    
    # 1. Wir laden jetzt EUER trainiertes Modell (das Gehirn aus Runde 1)
    # WICHTIG: Prüft in euren Ordnern, ob der Pfad exakt stimmt. 
    # Falls euer letzter Ordner z.B. brawli_cpu2 hieß, ändert das "brawli_cpu" hier ab!
    model = YOLO("runs/detect/brawli_cpu/weights/best.pt") 

    # 2. Training fortsetzen
    model.train(
        data="data.yaml",
        epochs=50,            # Wir geben ihm wieder 50 Runden für die neuen Bilder
        imgsz=640,
        batch=8,           
        device="cpu",         # Wir bleiben sicherheitshalber auf der CPU
        name="brawli_cpu_v2", # NEUER NAME: Speichert das neue Modell in einem neuen Ordner
        plots=True
    )

if __name__ == "__main__":
    start_training()
