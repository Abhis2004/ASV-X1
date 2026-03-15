from ultralytics import YOLO

def main():
    model = YOLO(r"C:\Users\araj8\Music\ASV-X1\intelligence\train\models\trained_drone_viz.pt")
    model.train(data=r"C:\Users\araj8\Music\ASV-X1\intelligence\train\yamls\add_drone.yaml", 
                      epochs=50,
                      imgsz=640)

if __name__ == "__main__":
    main()