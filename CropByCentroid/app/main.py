import cv2
import numpy as np
import torch

from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

def calculate_centroid(positions, direction_vectors):

    centroid_positions = []

    # 最初のフレームの重心を計算
    first_frame = positions[0]
    center_x = np.mean([player[0] for player in first_frame])
    center_y = np.mean([player[1] for player in first_frame])
    centroid_positions.append((center_x, center_y))
    
    for i in range(1, len(positions)):
        frame = positions[i]
        center_x = np.mean([player[0] for player in frame])
        center_y = np.mean([player[1] for player in frame])

        current_centroid = np.array([center_x, center_y])
        direction_vector = np.array(direction_vectors[i-1])

        direction_vector_adjusted = np.array([
            direction_vector[0] * 1,  # x軸の倍率
            direction_vector[1] * -10   # y軸の倍率
        ])

        corrected_centroid = current_centroid + direction_vector_adjusted
        centroid_positions.append(tuple(corrected_centroid))

    return centroid_positions

def calculate_direction_vector(positions):
    avg_positions = []
    for positions_by_frame in positions:
        avg_x = np.mean([player[0] for player in positions_by_frame])
        avg_y = np.mean([player[1] for player in positions_by_frame])
        avg_positions.append((avg_x, avg_y))

    direction_vectors = []

    for i in range(1, len(avg_positions)):
        prev_avg = np.array(avg_positions[i - 1])
        curr_avg = np.array(avg_positions[i])
        direction_vector = curr_avg - prev_avg
        direction_vectors.append(tuple(direction_vector))

    return direction_vectors


def smoothing(centroid_positions, sigma):
    x_positions = [x for x, y in centroid_positions]
    y_positions = [y for x, y in centroid_positions]

    smoothed_x = gaussian_filter1d(x_positions, sigma)
    smoothed_y = gaussian_filter1d(y_positions, sigma)

    smoothed_positions = np.vstack((smoothed_x, smoothed_y)).T

    return smoothed_positions


def add_xy_position(recognition, positions) -> None:

    boxes = recognition[0].boxes.xywh.tolist()
    classes = recognition[0].boxes.cls.tolist()

    xy_pairs = []
    for i, cls in enumerate(classes):
        if int(cls) == 0:
            xy_pairs.append((boxes[i][0], boxes[i][1]))
    
    positions.append(xy_pairs)

def plot_centroid(frame, position):
    x = position[0]
    y = position[1]
    
    color = (0, 0, 255)  # BGR形式で赤色
    radius = 4  # 点の半径
    thickness = -1  # 塗りつぶし

    cv2.circle(frame, (int(x),int(y)), radius, color, thickness)
    return frame

def crop(frame, position):
    x = position[0]
    y = position[1]

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    crop_frame = frame_pil.crop((x-270, y-270, x+270, y+270))

    crop_frame = np.array(crop_frame)
    crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_RGB2BGR)
    return crop_frame


def main():
    print(torch.cuda.is_available())  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO("./models/yolov10x.pt").to(device)

    cap = cv2.VideoCapture("app/data/movie.mp4")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    positions = []

    with tqdm(total=frame_count, desc="Processing Calculate Centroid", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                recognition = model.predict(frame, device=device, verbose=False)
                add_xy_position(recognition, positions)
                pbar.update(1)  # 進捗バーを1つ進める
            else:
                break
                
    cap.release()
    print(f"Total Frame: {frame_count}")
    print(f"len positions: {len(positions)}")
    
    direction_vectors = calculate_direction_vector(positions)
    centroids = calculate_centroid(positions, direction_vectors)

    smoothed_positions = smoothing(centroid_positions=centroids, sigma=3)

    cap = cv2.VideoCapture("app/data/movie.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    crop_frame_size = (540, 540)
    
    out = cv2.VideoWriter('app/output/crop_video/output_video.mp4', fourcc, fps, crop_frame_size)
    frame_index = 0
    with tqdm(total=frame_count, desc="Processing Crop", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                position = smoothed_positions[frame_index]
                frame = plot_centroid(frame, position)
                frame = crop(frame, position)
                
                out.write(frame)

                frame_index += 1
                pbar.update(1)
            else:
                break
    cap.release()
    out.release()

if __name__ == '__main__':
    main()
