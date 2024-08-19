#!/bin/bash

# Pythonスクリプトを実行
python app/main.py

# Pythonスクリプトの終了ステータスを確認
if [ $? -eq 0 ]; then
    echo "Pythonスクリプトが正常に実行されました。FFmpegを実行します。"
    
    # FFmpegコマンドを実行
    ffmpeg -framerate 25 -i ./app/output/crop_frames/frame_%05d.jpg -c:v libx264 -pix_fmt yuv420p ./app/output/crop_video/crop.mp4

    if [ $? -eq 0 ]; then
        echo "FFmpegが正常に実行されました。"
    else
        echo "FFmpegの実行中にエラーが発生しました。"
    fi
else
    echo "Pythonスクリプトの実行中にエラーが発生しました。FFmpegは実行されません。"
fi