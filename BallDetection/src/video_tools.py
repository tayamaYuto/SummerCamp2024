import cv2
import numpy as np

class VideoProcessor:
    def __init__(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def output_info(self):
        self.fourcc = cv2.VideoWriter_fourcc(*'avc1')
        return self.fourcc
        

class ImageProcessor:
    def plot_center(self, frame, x, y):

        color = (0, 0, 255)  # BGR形式で赤色
        radius = 2  # 点の半径
        thickness = -1  # 塗りつぶし

        cv2.circle(frame, (x, y), radius, color, thickness)
        return frame

    def crop(self, frame, center_x, center_y):
        width, height = 1280, 720
        crop_size = 270

        # x, yがはみ出した分だけ調整
        center_x = max(crop_size, min(center_x, width - crop_size))
        center_y = max(crop_size, min(center_y, height - crop_size))
        
        # フレームを切り取る範囲を決定し、int型に変換
        start_x = int(max(0, center_x - crop_size))
        end_x = int(min(width, center_x + crop_size))
        start_y = int(max(0, center_y - crop_size))
        end_y = int(min(height, center_y + crop_size))

        # フレームを切り取る
        crop_frame = frame[start_y:end_y, start_x:end_x]

        return crop_frame