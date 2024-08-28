import cv2
import numpy as np

class VideoProcessor:
    def __init__(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def output_info(self):
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return self.fourcc
        

class ImageProcessor:
    def plot_center(self, frame, x, y):

        color = (0, 0, 255)  # BGR形式で赤色
        radius = 2  # 点の半径
        thickness = -1  # 塗りつぶし

        cv2.circle(frame, (x, y), radius, color, thickness)
        return frame

    def crop(self, frame, x, y, crop_size=540):
        half_size = crop_size // 2
        
        # クロップ範囲を計算
        x_min = int(x - half_size)
        x_max = int(x + half_size)
        y_min = int(y - half_size)
        y_max = int(y + half_size)
        
        # フレームの境界を超えた場合の調整
        if x_min < 0 or x_max > frame.shape[1] or y_min < 0 or y_max > frame.shape[0]:
            # 必要なだけのパディングを計算
            top = max(0, -y_min)
            bottom = max(0, y_max - frame.shape[0])
            left = max(0, -x_min)
            right = max(0, x_max - frame.shape[1])
            
            # パディングを追加（ミラーリング）
            frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_REFLECT)
            
            # 新しいクロップ範囲を計算
            x_min += left
            x_max += left
            y_min += top
            y_max += top

        # クロップ範囲が 540x540 になるように切り取る
        crop_frame = frame[y_min:y_max, x_min:x_max]
        
        # クロップがcrop_sizeに満たない場合のサイズ調整
        if crop_frame.shape[0] < crop_size or crop_frame.shape[1] < crop_size:
            crop_frame = cv2.resize(crop_frame, (crop_size, crop_size))
            
        return crop_frame