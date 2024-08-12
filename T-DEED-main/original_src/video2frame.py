import os
import cv2
import argparse
from tqdm import tqdm

def video2frame(video_path, output_folder):
    # 動画ファイルを読み込む
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    with tqdm(total=total_frames, desc='Extracting frames', unit='frame') as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = os.path.join(output_folder, f'frame{frame_count}.jpg')
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    print(f'合計{frame_count}フレームが保存されました')

def main():

    parser = argparse.ArgumentParser(description='Video to Frame Extractor')
    parser.add_argument('--game_name', type=str, required=True, help='Game name in the format "YYYY-MM-DD - Team1 - Team2"')
    args = parser.parse_args()

    game_name = args.game_name
    video_path = f"/home/yuto/SummerCamp2024/T-DEED-main/data/soccernet/2019-2020/{game_name}/720p.mp4"
    output_folder = f"/home/yuto/SummerCamp2024/T-DEED-main/data/soccernet/frame_data/{game_name}"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video2frame(video_path=video_path, output_folder=output_folder)


if __name__ == "__main__":
    main()