# global
import os
import cv2
import numpy as np
from tqdm import tqdm


# local
from src.cost_minimize import CostMinimization
from src.video_tools import VideoProcessor, ImageProcessor
from src.get_change_frame import SceneDetector
from src.yolo_model import YoloModel
from src.logger_config import logger


def main():
    dir_path = "./input"
    files = os.listdir(dir_path)
    files = [i for i in files if i.endswith('.mp4') == True]


    for video in files:
        video_path = os.path.join(dir_path, video)
        video_processor = VideoProcessor(video_path)

        cap = video_processor.cap
        frame_count = video_processor.frame_count
        # cap = cv2.VideoCapture(video_path)
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.debug(f"frame count:{frame_count}")

        start_frame_list = []
        if not start_frame_list:
            start_frame_list = [0, frame_count]

        for i in range(len(start_frame_list) - 1):
            start_frame = start_frame_list[i]
            end_frame = start_frame_list[i + 1]
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            with tqdm(total=end_frame - start_frame, desc=f"Processing Interval {i+1}/{len(start_frame_list)-1}", unit="frame") as pbar:
                frame_index = start_frame
                
                while frame_index < end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(f"Failed to read frame at index {frame_index}")
                        break