from ultralytics import YOLO
import cv2

video_path = r"C:\Users\araj8\Downloads\follow.mp4"

model = YOLO(r"C:\Users\araj8\Music\ASV-X1\intelligence\train\models\trained_drone_viz.pt")
#model = YOLO(r"C:\Users\araj8\Music\ASV-X1\intelligence\train\yolo26n.pt")


selected_id = None
current_boxes = []  


def mouse_callback(event, x, y, flags, param):
    global selected_id

    if event == cv2.EVENT_LBUTTONDOWN:
        for box, track_id in current_boxes:
            x1, y1, x2, y2 = box

            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_id = track_id
                break


def track(video_path):

    global current_boxes, selected_id

    cap = cv2.VideoCapture(video_path)

    cv2.namedWindow("ByteTrack Tracking")
    cv2.setMouseCallback("ByteTrack Tracking", mouse_callback)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (640, 640))

        results = model.track(
            frame,
            tracker="bytetrack.yaml",
            persist=True,
            conf=0.1,
            imgsz=640,
        )

        result = results[0]

        current_boxes = []

        if result.boxes is not None and result.boxes.id is not None:

            boxes = result.boxes.xyxy
            ids = result.boxes.id
            confs = result.boxes.conf
            classes = result.boxes.cls

            for box, track_id, conf, cls in zip(boxes, ids, confs, classes):

                x1, y1, x2, y2 = map(int, box)

                current_boxes.append(((x1, y1, x2, y2), int(track_id)))

                label = f"ID:{int(track_id)} C:{int(cls)} {float(conf):.2f}"

                if int(track_id) == selected_id:
                    color = (0, 0, 255)   # red
                else:
                    color = (0, 255, 0)   # green

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # cv2.putText(
                #     frame,
                #     label,
                #     (x1, y1 - 10),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.6,
                #     color,
                #     2
                # )

        cv2.imshow("ByteTrack Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


track(video_path)