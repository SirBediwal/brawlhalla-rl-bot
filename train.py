from ultralytics import YOLO

def start_training():
    print("--- STARTE TRAINING (2 KLASSEN: ME vs ENEMY) ---")
    
    # 1. HIER euer Version 3 Modell eintragen! 
    # (Schaut nach, ob der Ordner brawli_cpu3 oder v3 heißt)
    model = YOLO("runs/detect/brawli_cpu_v3/weights/best.pt") 

    # 2. Training starten
    model.train(
        data="data.yaml",     # WICHTIG: Hier muss jetzt nc: 2 drinstehen!
        epochs=50,            
        imgsz=640,
        batch=8,           
        device="cpu",         
        name="brawli_2classes_v1", # Neuer Name, damit wir wissen, dass das hier das 2-Klassen-Modell ist
        plots=True
    )

if __name__ == "__main__":
    start_training()
