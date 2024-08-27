from ultralytics import YOLO


class YoloModel:
    def __init__(self, model_path="./models/best.pt"):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        self.model = YOLO(self.model_path)
        return self.model