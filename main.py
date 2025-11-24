from ultralytics import YOLO

model = YOLO("train17.pt")


results = model("0",show=True)  # predict realtime with webcam on an image


print("ok",results)
