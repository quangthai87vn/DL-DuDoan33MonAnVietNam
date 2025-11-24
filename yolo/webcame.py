from ultralytics import YOLO
model = YOLO("yolo_models/food_yolov11_best.pt")
results = model(
                "0",
                show=True,
                conf=0.6
                )  # predict realtime with webcam on an image
print("ok",results)