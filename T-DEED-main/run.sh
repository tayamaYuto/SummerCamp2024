#!/bin/bash

# Pythonスクリプトのパス
SCRIPT1="train_tdeed.py"
SCRIPT2="original_src/formated_json.py"

# スクリプト1に渡す引数
MODEL="SoccerNet"
AG=1

# スクリプト1を実行
echo "Running $SCRIPT1 with arguments --model $MODEL -ag $AG..."
python3 $SCRIPT1 --model $MODEL -ag $AG

# スクリプト1が成功したかチェック
if [ $? -eq 0 ]; then
    echo "$SCRIPT1 completed successfully."
    echo "Running $SCRIPT2..."
    python3 $SCRIPT2

    # スクリプト2の実行結果をチェック
    if [ $? -eq 0 ]; then
        echo "$SCRIPT2 completed successfully."
    else
        echo "Error: $SCRIPT2 failed."
    fi
else
    echo "Error: $SCRIPT1 failed. $SCRIPT2 will not be executed."
fi