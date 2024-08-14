import cv2

def get_total_frames(video_path):
    # ビデオファイルを読み込む
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # トータルフレーム数を取得する
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Total Frames: {total_frames}")

    # ビデオキャプチャを解放する
    cap.release()

if __name__ == "__main__":
    video_path = 'data/soccernet/2019-2020/2019-10-01 - Brentford - Bristol City/720p.mp4'  # ここにビデオファイルのパスを指定
    get_total_frames(video_path)