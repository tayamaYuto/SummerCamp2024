import os
import cv2
from tqdm import tqdm

def video2frame(video_path, output_folder):
    # 動画ファイルを読み込む
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 出力フォルダが存在しない場合は作成する
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
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
    video_path = "/home/yuto/SummerCamp2024/T-DEED-main/data/soccernet/West/720p.mp4"
    output_folder = "/home/yuto/SummerCamp2024/T-DEED-main/data/soccernet/data_folder/West"

    video2frame(video_path=video_path, output_folder=output_folder)


if __name__ == "__main__":
    main()