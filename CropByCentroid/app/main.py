import cv2
import numpy as np

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

def plot_centroid(frame, position, frame_index) -> None:
    x = position[0]
    y = position[1]
    
    color = (0, 0, 255)  # BGR形式で赤色
    radius = 4  # 点の半径
    thickness = -1  # 塗りつぶし

    cv2.circle(frame, (int(x),int(y)), radius, color, thickness)
    file_name = f"app/output/plot_frames/frame_{frame_index:05d}.jpg"

    cv2.imwrite(file_name, frame)

def crop(frame, position, frame_index) -> None:
    x = position[0]
    y = position[1]

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    crop_frame = frame_pil.crop((x-270, y-270, x+270, y+270))
    crop_frame.save(f'app/output/crop_frames/frame_{frame_index:05d}.jpg')


def main():
    model = YOLO("../models/yolov10x.pt")
    cap = cv2.VideoCapture("app/data/movie.mp4")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    positions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            recognition = model(frame)
            add_xy_position(recognition, positions)
        else:
            break
            
    cap.release()
    print(f"Total Frame: {frame_count}")
    print(f"len positions: {len(positions)}")
    
    direction_vectors = calculate_direction_vector(positions)
    centroids = calculate_centroid(positions, direction_vectors)

    smoothed_positions = smoothing(centroid_positions=centroids, sigma=3)

    cap = cv2.VideoCapture("app/data/movie.mp4")
    frame_index = 0
    with tqdm(total=frame_count) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                position = smoothed_positions[frame_index]
                plot_centroid(frame, position, frame_index)
                crop(frame, position, frame_index)

                frame_index += 1
                pbar.update(1)
            else:
                break
    cap.release()

if __name__ == '__main__':
    main()
