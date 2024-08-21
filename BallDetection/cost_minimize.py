import cv2
import os
import math
import numpy as np

from ultralytics import YOLO
from tqdm import tqdm


def plot_center(frame, center_x, center_y) :

    color = (0, 0, 255)  # BGR形式で赤色
    radius = 2  # 点の半径
    thickness = -1  # 塗りつぶし

    cv2.circle(frame, (center_x, center_y), radius, color, thickness)

    return frame

def calc_distance_between_nodes(bboxes):
    distances_between_nodes = []
    for i in range(len(bboxes) - 1):
        points1 = bboxes[i][:, :2]  # i番目のフレームの全ての(x, y)座標
        points2 = bboxes[i + 1][:, :2]  # i+1番目のフレームの全ての(x, y)座標
        
        # 各フレーム内の全てのノード間の距離を計算
        subset_distance = np.linalg.norm(points2[:, np.newaxis, :] - points1, axis=2)
        distances_between_nodes.append(subset_distance)
    
    return distances_between_nodes


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
        
        bboxes = []
        confs = []
        clss = []
        with tqdm(total=frame_count, desc="Processing Calculate Center", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    recognition = model.predict(frame, conf=0.01, verbose=False, classes = [1])


                    bbox = recognition[0].boxes.xywh.tolist()
                    if len(bbox) == 0:
                        previous_bbox = bboxes[-1].copy()
                        previous_bbox[:, :2] += 0.2
                        bbox = previous_bbox
                    bbox = np.array(bbox)
                    cls = recognition[0].boxes.cls.tolist()
                    conf = recognition[0].boxes.conf.tolist()

                    bboxes.append(bbox)
                    clss.append(cls)
                    confs.append(conf)

                    frame_index += 1
                    pbar.update(1)

                else:
                    break

        cap.release()
        print(frame_count)
        print(len(clss))
        print(bboxes[3])
        print(type(bboxes[3][0]))
        print(clss[3])
        print(confs[3])

        distances = calc_distance_between_nodes(bboxes)
        print(len(bboxes) - 1)
        print(len(distances))

if __name__ == '__main__':
    main()
