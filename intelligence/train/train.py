from ultralytics import YOLO

def main():
    model = YOLO("models/yolo26n.pt")
    results = model.train(data="VisDrone.yaml", epochs=50, imgsz=640)

if __name__ == "__main__":
    main()