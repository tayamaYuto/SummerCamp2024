import cv2

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