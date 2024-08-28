# global
import time
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
        basename = os.path.basename(video)
        video_path = os.path.join(dir_path, video)
        scene_detector = SceneDetector(video_path, 30)
        video_processor = VideoProcessor(video_path)

        start_frame_list = scene_detector.get_scene_change_frame()
        cap = video_processor.cap
        frame_count = video_processor.frame_count
        logger.debug(f"frame count:{frame_count}")

        total_out_bboxes = []

        if start_frame_list:
            start_frame_list.append(frame_count)
        else:
            start_frame_list = [0, frame_count]

        for i in range(len(start_frame_list) - 1):
            start_frame = start_frame_list[i]
            end_frame = start_frame_list[i + 1]

            
            model = YoloModel().load_model()

            bboxes = []
            confs = []
            clss = []
            with tqdm(total=end_frame - start_frame, desc=f"Processing Interval {i+1}/{len(start_frame_list)-1}", unit="frame") as pbar:
                frame_index = start_frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                while frame_index < end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(f"Failed to read frame at index {frame_index}")
                        break
                    recognition = model.predict(frame, conf=0.001, verbose=False, classes=[1])

                    bbox = recognition[0].boxes.xywh.tolist()
                    if len(bbox) == 0:
                        bbox = [[0, 0, 0, 0]]
                        bbox = np.array(bbox)
                        bboxes.append(bbox)
                        frame_index += 1
                        pbar.update(1)
                        continue

                    bbox = np.array(bbox)
                    cls = recognition[0].boxes.cls.tolist()
                    conf = recognition[0].boxes.conf.tolist()

                    bboxes.append(bbox)
                    clss.append(cls)
                    confs.append(conf)

                    frame_index += 1
                    pbar.update(1)

            logger.debug(f"Fist input box type:{type(bboxes[0])}")
            minimization_processor = CostMinimization(confs, bboxes)
            output_bboxes = minimization_processor.process_cost_minimization()
            total_out_bboxes.append(output_bboxes)
        cap.release()

        image_processor = ImageProcessor()
        out_video_processor = VideoProcessor(video_path)
        cap = out_video_processor.cap
        fps = out_video_processor.fps
        fourcc = out_video_processor.output_info()

        output_folder = "./output"
        output_path = os.path.join(output_folder, "crop_"+ basename)
        out = cv2.VideoWriter(output_path, fourcc, fps, (540, 540), isColor=True)

        total_out_bboxes = np.concatenate(total_out_bboxes)
        logger.debug(len(total_out_bboxes))
        assert len(total_out_bboxes) == frame_count, "Error: total_out_bboxes length does not match the frame count."

        frame_index = 0
        out_count = 0
        output_dir = "output/output_frames"  # 保存先ディレクトリを指定
        os.makedirs(output_dir, exist_ok=True)
        height, width = 540, 540
        with tqdm(total=frame_count) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                x = total_out_bboxes[frame_index][0]
                y = total_out_bboxes[frame_index][1]

                cropped_frame = image_processor.crop(frame, x, y)

                if cropped_frame.shape[1] == width and cropped_frame.shape[0] == height:
                    out_count += 1

                out.write(cropped_frame)
                frame_index += 1
                pbar.update(1)

        logger.info(f"out count:{out_count}")
        cap.release()
        out.release()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"処理が終了しました。処理時間: {elapsed_time:.2f}秒")



