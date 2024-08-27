import numpy as np

class LinearInterpolation:
    def interpolate_zeros(self, bboxes):
        # x, y のインデックス
        zero_indices = np.where((bboxes[:, 0] == 0) & (bboxes[:, 1] == 0))[0]
        
        if len(zero_indices) == 0:
            return bboxes

        # 連続したゼロのインデックスグループを特定
        groups = self._identify_zero_groups(zero_indices)
        
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
    
    def _identify_zero_groups(self, zero_indices):
        return np.split(zero_indices, np.where(np.diff(zero_indices) != 1)[0] + 1)