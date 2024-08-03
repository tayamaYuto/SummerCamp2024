import argparse
import cv2
import os

def video_to_image_all(input_path, output_path='./data/output', basename='f', extension='png',frame_num=10) -> None:
    """Save all frames of a video to images.

    Args:
        input_path (str): input_path of the video.
        output_path (str, optional): output_path of the images. Defaults to '../data/output'.
        basename (str, optional): basename for the saved images. Defaults to 'frame'.
        extension (str, optional): extension for the saved images. Defaults to 'jpg'.
        frame_num (int,optional): length of frame to procces. Defaults to .
    """
    print(input_path)
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        return


    #入力ファイルの拡張子を除いた名前を取得
    input_file_name = os.path.basename(input_path)
    input_file_name =os.path.splitext(input_file_name)[0]
    print(f"input_file_name: {input_file_name}")
    output_path = os.path.join(output_path, input_file_name)
    print(f"output_path: {output_path}")

    os.makedirs(output_path, exist_ok=True)



    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    digit = len(str(frame_count))


    n = 0
    while True:
        ret, frame = cap.read()
        if ret and n < frame_num:
            file_name = '{}_{}_{}.{}'.format(input_file_name, basename, str(n).zfill(digit), extension)
            full_path = os.path.join(output_path, file_name)
            cv2.imwrite(full_path, frame)
            n += 1

        elif  n > frame_num:
            break

        else:
            break

def main():
    #実行コード
    # python video2image.py ../data/input/img_3822.mov
    print(os.getcwd())

    input_path = os.getcwd() + "/data/input/720p.mp4"

    parser = argparse.ArgumentParser(description='Save all frames of a video to images.')
    # parser.add_argument('input_path', type=str,help='Path to the input video folder.')
    parser.add_argument('-o','--output_path', type=str, default="./data/output",  help='Directory to save the images. Defaoult is "./data/output"')
    parser.add_argument('-b', '--basename', type=str, default='f', help='Basename for the saved images. Default is "f".')
    parser.add_argument('-ie', '--img_extension', type=str, default='png', help='img_extension for the saved images. Default is "png".')
    parser.add_argument('-fn', '--frame_num', type=int, default=10, help='length of frame to process')

    args = parser.parse_args()

    #入力フォルダからファイル名を取得
    # file_path = args.input_path
    file_path = input_path

    video_to_image_all(file_path, args.output_path, args.basename, args.img_extension, args.frame_num)

if __name__ == '__main__':
    main()
