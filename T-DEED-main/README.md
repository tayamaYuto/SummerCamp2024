# 合宿用T-DEED動かし方

### 動作環境
RTX 3060
VRAM 12GB
WSL2 20.04
CUDA 12.1
Driver Version: 536.23

### パッケージインストール

仮想環境の作成
```bash
python -m venv .env  # .envは任意の名前
```

仮想環境に入る
```bash
source .env/bin/activate
```

必要パッケージのinstall
```bash
pip install -r requirements.txt
```

### データの配置と前準備
映像データ
data/soccernet/2019-2020/各試合名のフォルダ/720p.mp4 + labels-ball.json

この配置をしてから以下を実行（T-DEED実行に必要なkeyとvalueを追加）
```bash
python original_src/insert_video_info.py
```
実行後、jsonデータがこのようになっていたらOKです
```json
{
    "UrlLocal": "2019-10-01 - Blackburn Rovers - Nottingham Forest",
    "fps": 25,
    "num_frames": "142887",
    "annotations": [
        {
            "gameTime": "1 - 00:00",
            "label": "PASS",
            "position": "680",
            "team": "away",
            "visibility": "visible"
        }
    ]
}
```
更新されたjsonファイルを以下のフォルダで配置

アノテーションデータ
data/soccernet/2019-2020-json/(任意の名前).json, (任意の名前2).json, ...

jsonデータ配置後、以下のコマンドを実施
このスクリプトの実行で複数試合のjsonが1つのjsonファイルになります
```bash
python original_src/concat_json.py
```
次にsplit_json_dataスクリプトを実行して、訓練、検証、テストに分割

```bash
python original_src/split_json_data.py
```

最後に動画をフレームに分割（1つ1つ実行する必要があります）

```bash
python original_src/frame2video.py --game_name "2019-10-01 - Blackburn Rovers - Nottingham Forest"
```

実行後、data/soccernet/frame_folder/試合名のフォルダにフレームデータが入っています（1動画あたりおおよそ15万フレーム：30GB）


### 学習

```bash
python train_tdeed.py --model SoccerNet
```

学習時の設定はconfig/SoccerNet/SoccerNet.jsonで指定可能



