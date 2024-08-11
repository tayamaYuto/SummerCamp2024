import json
import random

def main():
    with open('data/soccernet/Labels-ball.json', 'r') as file:
        data = json.load(file)
    
    train_ratio = 0.8
    validation_ratio = 0.1
    
    train_data = []
    validation_data = []
    test_data = []

    for item in data:
        
        annotations = item['annotations']

        indices = list(range(len(annotations)))
        random.shuffle(indices)

        train_split_index = int(len(indices) * train_ratio)
        validation_split_index = int(len(indices) * (train_ratio + validation_ratio))

        train_indices = sorted(indices[:train_split_index])
        validation_indices = sorted(indices[train_split_index:validation_split_index])
        test_indices = sorted(indices[validation_split_index:])

        train_annotations = [annotations[i] for i in train_indices]
        validation_annotations = [annotations[i] for i in validation_indices]
        test_annotations = [annotations[i] for i in test_indices]
        
        game = item['UrlLocal']
        print(f'Process: {game}')
        train_item = {
            "UrlLocal": item['UrlLocal'],
            "fps": item['fps'],
            "num_frames": item['num_frames'],
            "annotations": train_annotations
        }

        validation_item = {
            "UrlLocal": item['UrlLocal'],
            "fps": item['fps'],
            "num_frames": item['num_frames'],
            "annotations": validation_annotations
        }

        test_item = {
            "UrlLocal": item['UrlLocal'],
            "fps": item['fps'],
            "num_frames": item['num_frames'],
            "annotations": test_annotations
        }

        train_data.append(train_item)
        validation_data.append(validation_item)
        test_data.append(test_item)

    with open('data/soccernet/train.json', 'w') as train_file:
        json.dump(train_data, train_file, indent=4)

    with open('data/soccernet/val.json', 'w') as validation_file:
        json.dump(validation_data, validation_file, indent=4)

    with open('data/soccernet/test.json', 'w') as test_file:
        json.dump(test_data, test_file, indent=4)
    
    print('Finish Split!')


if __name__ == "__main__":
    main()