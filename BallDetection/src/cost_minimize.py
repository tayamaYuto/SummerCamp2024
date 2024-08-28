import numpy as np
import heapq

from scipy.ndimage import gaussian_filter

from src.linear_interpolation import LinearInterpolation
from src.logger_config import logger



class CostMinimization:
    def __init__(self, confs, bboxes):
        self.confs = confs
        self.bboxes = bboxes
        self.linear_interpolation = LinearInterpolation()

    def _get_null_frames(self):
        
        null_bbox = np.array([[0, 0, 0, 0]])
        null_frame_indices = [i for i, bbox in enumerate(self.bboxes) if np.array_equal(bbox, null_bbox)]
        filtered_bboxes = [bbox for bbox in self.bboxes if not np.array_equal(bbox, null_bbox)]

        return null_frame_indices, filtered_bboxes
    
    def _establish_ball_position(self, position):
        candidates = self.confs[position]
        max_value = max(candidates)
        max_index = candidates.index(max_value)
        return self.bboxes[position][max_index]
    
    def _calc_distance_between_nodes(self):
        distances_between_nodes = []
        for i in range(len(self.bboxes) - 1):
            points1 = self.bboxes[i][:, :2]  # i番目のフレームの全ての(x, y)座標
            points2 = self.bboxes[i + 1][:, :2]  # i+1番目のフレームの全ての(x, y)座標
            
            # 各フレーム内の全てのノード間の距離を計算
            subset_distance = np.linalg.norm(points2[:, np.newaxis, :] - points1, axis=2)
            distances_between_nodes.append(subset_distance)
        
        return distances_between_nodes

    def _count_total_elements(self, distances):
        total_elements = 0
        
        for distance in distances:
            total_elements += distance.shape[1]
        
        return total_elements

    def _create_edges(self, _distances):
        edges = [distance.T.tolist() for distance in _distances]

        fixed_value = 0
        result = []
        node_offset_list = []
        for node_edge in edges:
            fixed_value += len(node_edge)
            node_offset_list.append(fixed_value)
            for dis in node_edge:
                node_number_and_value = [[i + fixed_value, value] for i, value in enumerate(dis)]
                result.append(node_number_and_value)
        result.append([])
        return result, node_offset_list

    def _dijkstra(self, edges, num_node, Goal):
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
    
    def process_cost_minimization(self):
        crop_bboxes = np.array([])
        logger.debug(len(self.bboxes))
        null_frames, not_null_frames = self._get_null_frames()
        self.bboxes = not_null_frames

        # 始点と終点を確立
        self.bboxes[0] = np.array([self._establish_ball_position(position=0)])
        self.bboxes[-1] = np.array([self._establish_ball_position(position=-1)])

        distances = self._calc_distance_between_nodes()
        total_nodes = self._count_total_elements(distances)
        edges, node_offset_list = self._create_edges(distances)

        # ダイクストラ法計算
        opt_root, _ = self._dijkstra(edges, total_nodes + 1, total_nodes)

        bbox_index = self._get_opt_index(opt_root, node_offset_list)
        bbox_index = np.insert(bbox_index, 0, 0)
        bbox_index = bbox_index.tolist()

        crop_bboxes = self._get_bbox(bbox_index)

        values_to_insert = np.array([0, 0, 0, 0])

        for index in null_frames:
            crop_bboxes = np.insert(crop_bboxes, index, values_to_insert, axis=0)
        logger.debug(f"type crop_bboxes: {type(crop_bboxes[0])}")
        logger.debug(f"shape of crop bboxes: {crop_bboxes.shape}")
        crop_bboxes = self.linear_interpolation.interpolate_zeros(crop_bboxes)
        bboxes_smoothed = gaussian_filter(crop_bboxes.astype(float), sigma=0.5)
        return bboxes_smoothed
    
    def _get_opt_index(self, opt_root, node_offset_list):
        node_numbers = np.array(opt_root)
        fix_number = np.array(node_offset_list)

        bbox_index = node_numbers[1: len(fix_number) + 1] - fix_number
        return bbox_index
    
    def _get_bbox(self, bbox_index):
        result = []
        logger.debug(f"Input box type :{self.bboxes[0][0]}")
        for index, bbox in zip(bbox_index, self.bboxes):
            result.append(bbox[index])
        result = np.array(result)
        return result
