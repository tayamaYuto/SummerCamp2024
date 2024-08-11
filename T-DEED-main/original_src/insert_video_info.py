import os
import cv2
import json
from collections import OrderedDict

def get_folder_names(directory_path):
    try:
        folder_names = [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]
        return folder_names

    except Exception as e:
        print(f"フォルダ名取得失敗-原因:{e}")
        return []

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return fps, num_frames

def insert_info_to_json(json_path, fps, num_frames, game) -> None:
    with open(json_path, 'r') as f:
        data = json.load(f)

    # デフォルトのときはこれ
    # url_local_value = data["UrlLocal"]
    # extracted_value = url_local_value.split('/')[-1]
    
    ordered_data = OrderedDict()
    ordered_data["UrlLocal"] = data["UrlLocal"]
    ordered_data["fps"] = int(fps)
    ordered_data["num_frames"] = f"{num_frames}"
    ordered_data["annotations"] = data["annotations"]

    output_path = f"data/soccernet/2019-2020-json/{game}.json"
    with open(output_path, 'w') as f:
        json.dump(ordered_data, f, indent=4)

def main():
    directory_path = "data/soccernet/2019-2020"
    game_names = get_folder_names(directory_path)

    for game in game_names:
        video_path = f"{directory_path}/{game}/720p.mp4"
        fps, num_frames = get_video_info(video_path)

        json_path = f"{directory_path}/{game}/Labels-ball.json"
        insert_info_to_json(json_path, fps, num_frames, game)
        print(f"Finish Insert! game name: {game}")
    print("Finish All Game!!")


if __name__ == "__main__":
    main()

