import json
import glob

def main():
    json_files_path = "data/soccernet/2019-2020-json/*.json"
    json_list = [json.load(open(file, 'r')) for file in glob.glob(json_files_path)]

    output_path = "data/soccernet/Labels-ball.json"
    with open(output_path, 'w') as outfile:
        json.dump(json_list, outfile, indent=4)
    
    print("Finish Concat Json Files!")


if __name__ == "__main__":
    main()
