import json


def main():
    input_file_path = 'data/save/soccernet/SoccerNet-1/pred-test.json'
    output_file_path = 'data/save/soccernet/SoccerNet-1/pred-test-formatted.json'

    # JSONデータをファイルから読み込み
    with open(input_file_path, 'r') as infile:
        data = json.load(infile)

    with open(output_file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4, ensure_ascii=False)
    
    print(f"Formatted JSON has been written to {output_file_path}")


if __name__ == '__main__':
    main()