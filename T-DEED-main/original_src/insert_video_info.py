import os
import cv2

def get_folder_names(directory_path):
    try:
        folder_names = [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]
        return folder_names

    except Exception as e:
        print(f"フォルダ名取得失敗-原因:{e}")
        return []
    
def get_video_path(folder_paths):
    video_paths = []
    for folder_path in folder_paths:
        for file in os.listdir(folder_path):
            if file.endswitch('.mp4'):
                video_paths.append(os.path.join(folder_path, file))
    return video_paths

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    game_name = video_path.split('/')[-2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return game_name, fps, num_frames

def insert_info_to_json():
    return

def main():

