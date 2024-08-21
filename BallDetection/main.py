import cv2
import numpy as np
import os
import torch

from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm

def calcuralte_center(boxes, classes, confs):
    center_x = False
    center_y = False
    max_conf = 0
    # center_x,center_y=640,360
    for i,cls in enumerate(classes):
        if cls == 1:
            if confs[i] > max_conf:
                max_conf = confs[i]
                center_x, center_y = int(boxes[i][0]), int(boxes[i][1])

    return center_x, center_y


def liner_suppliment_center(center_list, frame_num):
    # x=640
    # y=360
    # index=0
    dx = 0
    dy = 0
    d_frame = 0

    x_stan = -1
    y_stan = -1
    index_stan = -1

    for center_info in center_list:
        #1フレーム目は例外処理
        if center_info[0] == 0:
            if center_info[1] != False:
                x_stan = center_info[1]
                y_stan = center_info[2]
                index_stan = 0
            
            elif center_info[1] == False:
                x_stan = 640
                y_stan = 360
                index_stan = 0
                center_info[1] = x_stan
                center_info[2] = y_stan


        #線形補完を実装
        elif center_info[1] != False :
            d_frame = center_info[0] - index_stan
            dx = (center_info[1] - x_stan) / d_frame
            dy = (center_info[2] - y_stan) / d_frame

            for i in range(d_frame):
                center_list[index_stan + i][1] = int(center_list[index_stan][1] + i * dx)
                center_list[index_stan + i][2] = int(center_list[index_stan][2] + i * dy)
        
            x_stan = center_info[1]
            y_stan = center_info[2]
            index_stan = center_info[0]
        

        #最終フレームで検知がなかった場合も例外処理
        elif center_info[0] == frame_num and center_info[1] == False :
            d_frame = center_info[0] - index_stan
            dx = (640-x_stan) / d_frame
            dy = (360-y_stan) / d_frame

            for j in range(d_frame + 1):
                center_list[index_stan + j][1] = int(center_list[index_stan][1] + j * dx)
                center_list[index_stan + j][2] = int(center_list[index_stan][2] + j * dy)


    return center_list

def crop(frame, center_x, center_y):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    crop_frame = frame_pil.crop((center_x-270, center_y-270, center_x+270, center_y+270))

    crop_frame = np.array(crop_frame)
    crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_RGB2BGR)
    
    return crop_frame


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

    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO("./models/best.pt").to(device)

    for file in files:
        basename = os.path.basename(file)
        file_path = os.path.join(dir_path, file)

        cap = cv2.VideoCapture(file_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        center_list=[]
        frame_index=0

        with tqdm(total=frame_count, desc="Processing Calculate Center", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    recognition = model.predict(frame, device=device, verbose=False)


                    boxes =   recognition[0].boxes.xywh.tolist()
                    classes = recognition[0].boxes.cls.tolist()
                    confs = recognition[0].boxes.conf.tolist()

                    x, y = calcuralte_center(boxes, classes, confs)
                    center_list.append([frame_index, x, y])
                
                    frame_index += 1
                    pbar.update(1) 

                else:
                    break

        cap.release()
        
        #線形補完によって各フレームでのクロップ画像の中心点を決定する
        center_list = liner_suppliment_center(center_list, frame_index-1)

        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"fps:{fps}")
        

        output_folder = "./output"
        output_path = os.path.join(output_folder, "crop_"+ basename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (540,540), isColor=True)

        frame_index = 0
        with tqdm(total=frame_count) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    x, y = center_list[frame_index][1],center_list[frame_index][2]
                    
                    out.write(crop(frame, x, y))
            
                    frame_index += 1
                    pbar.update(1)
                else:
                    break
        cap.release()
        out.release()




if __name__ == '__main__':
    main()
