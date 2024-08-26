from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector

from scenedetect import open_video


class SceneDetector:
    def __init__(self, video_path, threshold):
        self.video_path = video_path
        self.threshold = threshold

    def get_scene_change_frame(self, min_scene_len=15):
        video = open_video(self.video_path)
        scene_manager = SceneManager()

        # ContentDetectorを閾値、最小シーン長と共に追加
        #threshold: 変化検出の閾値。この値よりも大きな変化がある場合にシーンとして検出される。
        #min_scene_len: シーンとして検出する最小のフレーム数。この値よりも短いシーンは検出されない。
        scene_manager.add_detector(ContentDetector(threshold=self.threshold, min_scene_len=min_scene_len ))

        # ビデオとシーンマネージャを関連付け
        scene_manager.detect_scenes(frame_source = video)

        # 検出されたシーンのリストを取得
        scene_list = scene_manager.get_scene_list()

        start_frame_list = []
        for i, scene in enumerate(scene_list):
            start_frame = scene[0].get_frames()
            start_frame_list.append(start_frame)

        return start_frame_list