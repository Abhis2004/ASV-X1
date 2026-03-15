from ultralytics import YOLO

def main():
    model = YOLO(r"C:\Users\araj8\Music\ASV-X1\intelligence\train\models\yolo26s.pt")
    model.train(data="VisDrone.yaml", 
                epochs=100, 
                imgsz=640,
                device='cuda',
                batch=-1)

if __name__ == "__main__":
    main()