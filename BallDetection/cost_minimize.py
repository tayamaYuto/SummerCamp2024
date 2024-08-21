import cv2
import numpy as np
import os

from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm


def plot_center(frame, center_x, center_y) :

    color = (0, 0, 255)  # BGR形式で赤色
    radius = 2  # 点の半径
    thickness = -1  # 塗りつぶし

    cv2.circle(frame, (center_x, center_y), radius, color, thickness)

    return frame



def main():
    dir_path = "./input"
    files = os.listdir(dir_path)
    files = [i for i in files if i.endswith('.mp4') == True]

    model = YOLO("./models/best.pt")

    for file in files:
        basename = os.path.basename(file)
        file_path = os.path.join(dir_path, file)

        cap = cv2.VideoCapture(file_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_index=0

        with tqdm(total=frame_count, desc="Processing Calculate Center", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    recognition = model.predict(frame, conf=0.01, verbose=False, classes = [1])


                    boxes =   recognition[0].boxes.xywh.tolist()
                    classes = recognition[0].boxes.cls.tolist()
                    confs = recognition[0].boxes.conf.tolist()

                    frame_index += 1
                    pbar.update(1)

                else:
                    break

        cap.release()
        print(classes)
        print(confs)

if __name__ == '__main__':
    main()
