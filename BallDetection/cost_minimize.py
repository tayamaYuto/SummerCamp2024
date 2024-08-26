import cv2
import os
import numpy as np
import heapq

from scipy.ndimage import gaussian_filter
from ultralytics import YOLO
from tqdm import tqdm


class VideoProcessor:
    def __init__(self, file_path):
        self.cap = cv2.VideoCapture(file_path)

    def video_info(self):
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
    
    def output_info(self):
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

class ImageProcessor:
    def plot_center(self, frame, x, y):

        color = (0, 0, 255)  # BGR形式で赤色
        radius = 2  # 点の半径
        thickness = -1  # 塗りつぶし

        cv2.circle(frame, (x, y), radius, color, thickness)
        return frame

    def crop(self, frame, x, y, crop_size=540):
        half_size = crop_size // 2
        
        x_min = int(max(0, x - half_size))
        x_max = int(min(frame.shape[1], x + half_size))
        
        y_min = int(max(0, y - half_size))
        y_max = int(min(frame.shape[0], y + half_size))
        
        # クロップ範囲が crop_size に満たない場合の調整
        if x_max - x_min < crop_size:
            x_min = max(0, x_max - crop_size)
        if y_max - y_min < crop_size:
            y_min = max(0, y_max - crop_size)
        
        crop_frame = frame[y_min:y_max, x_min:x_max]
        
        return crop_frame
    
class CostMinimization:
    def __init__(self):
        return None


def interpolate_zeros(bboxes):
    # x, y のインデックス
    zero_indices = np.where((bboxes[:, 0] == 0) & (bboxes[:, 1] == 0))[0]
    
    if len(zero_indices) == 0:
        return bboxes

    # 連続したゼロのインデックスグループを特定
    groups = np.split(zero_indices, np.where(np.diff(zero_indices) != 1)[0] + 1)
    
    for group in groups:
        start_index = group[0]
        end_index = group[-1]
        
        if start_index > 0 and end_index < len(bboxes) - 1:
            # 線形補完の対象
            prev = bboxes[start_index - 1, :2]
            next = bboxes[end_index + 1, :2]
            interpolated_values = np.linspace(prev, next, len(group) + 2)[1:-1]
            bboxes[group, :2] = interpolated_values
        elif start_index == 0:
            # 最初の要素の場合、次の要素をコピー
            bboxes[group, :2] = bboxes[end_index + 1, :2]
        elif end_index == len(bboxes) - 1:
            # 最後の要素の場合、前の要素をコピー
            bboxes[group, :2] = bboxes[start_index - 1, :2]
    
    return bboxes


def plot_center(frame, center_x, center_y) :

    color = (0, 0, 255)  # BGR形式で赤色
    radius = 2  # 点の半径
    thickness = -1  # 塗りつぶし

    cv2.circle(frame, (center_x, center_y), radius, color, thickness)

    return frame

def establish_ball_position(confs, bboxes, position):
    candidates = confs[position]
    max_value = max(candidates)
    max_index = candidates.index(max_value)
    return bboxes[position][max_index]


def get_null_frames(bboxes):
    
    null_bbox = np.array([[0, 0, 0, 0]])
    null_frame_indices = [i for i, bbox in enumerate(bboxes) if np.array_equal(bbox, null_bbox)]
    filtered_bboxes = [bbox for bbox in bboxes if not np.array_equal(bbox, null_bbox)]

    return null_frame_indices, filtered_bboxes

def calc_distance_between_nodes(bboxes):
    distances_between_nodes = []
    for i in range(len(bboxes) - 1):
        points1 = bboxes[i][:, :2]  # i番目のフレームの全ての(x, y)座標
        points2 = bboxes[i + 1][:, :2]  # i+1番目のフレームの全ての(x, y)座標
        
        # 各フレーム内の全てのノード間の距離を計算
        subset_distance = np.linalg.norm(points2[:, np.newaxis, :] - points1, axis=2)
        distances_between_nodes.append(subset_distance)
    
    return distances_between_nodes

def create_edges(distances):
    edges = [distance.T.tolist() for distance in distances]

    fixed_value = 0
    result = []
    fixed_value_list = []
    for node_edge in edges:
        fixed_value += len(node_edge)
        fixed_value_list.append(fixed_value)
        for dis in node_edge:
            node_number_and_value = [[i + fixed_value, value] for i, value in enumerate(dis)]
            result.append(node_number_and_value)
    result.append([])
    return result, fixed_value_list

def count_total_elements(distances):
    total_elements = 0
    
    for distance in distances:
        total_elements += distance.shape[1]
    
    return total_elements

def get_opt_index(opt_root, fixed_value_list):
    node_numbers = np.array(opt_root)
    fix_number = np.array(fixed_value_list)

    bbox_index = node_numbers[1: len(fix_number) + 1] - fix_number
    return bbox_index

def get_bbox(bbox_index, bboxes):
    result = []
    for index, bbox in zip(bbox_index, bboxes):
        result.append(bbox[index])
    
    return result


def dijkstra(edges, num_node, Goal):
    """ 経路の表現
            [終点, 辺の値]
            A, B, C, D, ... → 0, 1, 2, ...とする """
    node = [float('inf')] * num_node    #スタート地点以外の値は∞で初期化
    node[0] = 0     #スタートは0で初期化

    node_name = []
    heapq.heappush(node_name, [0, [0]])

    while len(node_name) > 0:
        #ヒープから取り出し
        _, min_point = heapq.heappop(node_name)
        last = min_point[-1]
        if last == Goal:
            return min_point, node  #道順とコストを出力させている
        
        #経路の要素を各変数に格納することで，視覚的に見やすくする
        for factor in edges[last]:
            goal = factor[0]   #終点
            cost  = factor[1]   #コスト

            #更新条件
            if node[last] + cost < node[goal]:
                node[goal] = node[last] + cost     #更新
                #ヒープに登録
                heapq.heappush(node_name, [node[last] + cost, min_point + [goal]])

    return []

def crop(frame, x, y, crop_size=540):
    half_size = crop_size // 2
    
    x_min = int(max(0, x - half_size))
    x_max = int(min(frame.shape[1], x + half_size))
    
    y_min = int(max(0, y - half_size))
    y_max = int(min(frame.shape[0], y + half_size))
    
    # クロップ範囲が crop_size に満たない場合の調整
    if x_max - x_min < crop_size:
        x_min = max(0, x_max - crop_size)
    if y_max - y_min < crop_size:
        y_min = max(0, y_max - crop_size)
    
    crop_frame = frame[y_min:y_max, x_min:x_max]
    
    return crop_frame

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
                    recognition = model.predict(frame, conf=0.001, verbose=False, classes = [1])


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

                else:
                    break

        cap.release()
        
        null_frames, bboxes = get_null_frames(bboxes)
        print(null_frames)

        bboxes[0] = np.array([establish_ball_position(confs, bboxes, 0)])
        bboxes[-1] = np.array([establish_ball_position(confs, bboxes, -1)])
        distances = calc_distance_between_nodes(bboxes)

        total_nodes = count_total_elements(distances)
        edges, fixed_value_list= create_edges(distances)

        # with open('output.txt', 'w') as f:
        #     json.dump(edges, f, indent=2)

        opt_root, _ = dijkstra(edges, total_nodes + 1, total_nodes)

        bbox_index = get_opt_index(opt_root, fixed_value_list)
        bbox_index = np.insert(bbox_index, 0, 0)
        bbox_index = bbox_index.tolist()

        crop_bboxes = get_bbox(bbox_index, bboxes)
        

        values_to_insert = [0, 0, 0, 0]

        for index in null_frames:
            crop_bboxes = np.insert(crop_bboxes, index, values_to_insert, axis=0)
        
        crop_bboxes = interpolate_zeros(crop_bboxes)
        bboxes_smoothed = gaussian_filter(crop_bboxes.astype(float), sigma=0.5)

        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"fps:{fps}")
        

        output_folder = "./output"
        output_path = os.path.join(output_folder, "crop_"+ basename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (540, 540), isColor=True)

        frame_index = 0
        with tqdm(total=frame_count) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    x = bboxes_smoothed[frame_index][0]
                    y = bboxes_smoothed[frame_index][1]
                    
                    out.write(crop(frame, x, y))
            
                    frame_index += 1
                    pbar.update(1)
                else:
                    break
        cap.release()
        out.release()





if __name__ == '__main__':
    main()
