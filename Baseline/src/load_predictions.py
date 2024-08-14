from pathlib import Path
from tqdm import tqdm
import json
import numpy as np

classes = ['PASS', 'DRIVE', 'HEADER', 'HIGH PASS', 'OUT', 'CROSS', 'THROW IN',
           'SHOT', 'BALL PLAYER BLOCK', 'PLAYER SUCCESSFUL TACKLE', 'FREE KICK',
           'GOAL']

class2target: dict[str, int] = {cls: trg for trg, cls in enumerate(classes)}


def load_video_predictions(game_prediction_dir: Path):
    raw_predictions_path = f"{game_prediction_dir}/1_raw_predictions.npz"
    raw_predictions_npz = np.load(str(raw_predictions_path))
    frame_indexes = raw_predictions_npz["frame_indexes"]
    raw_predictions = raw_predictions_npz["raw_predictions"]
    print(f"Total Frame: {len(raw_predictions)}")
    print(frame_indexes)

    frame_data = []

    for frame_index, prediction in tqdm(zip(frame_indexes, raw_predictions), total=len(frame_indexes), desc="Processing frames"):
        frame_info = {
                    "frame": int(frame_index),
                    "confidence": {cls: float(prediction[class2target[cls]]) for cls in classes}
                }
        frame_data.append(frame_info)

    return frame_data

def save_frame_data_as_json(frame_data, output_file: Path):
    with open(output_file, 'w') as f:
        json.dump(frame_data, f, indent=4)

def main():
    game_prediction_path = "data/predictions"
    frame_data = load_video_predictions(game_prediction_path)

    output_file = Path("pred_actions.json")
    save_frame_data_as_json(frame_data, output_file)


if __name__ == "__main__":
    main()

