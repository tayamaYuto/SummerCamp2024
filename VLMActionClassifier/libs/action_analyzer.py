import ollama
import glob
import os
import time

def action_analyzer_prompt():
    prompt = """
You have the ability to analyze player actions from soccer match footage. Please watch the soccer match footage and analyze which action from the following list the player is performing. If the action does not fall under any of the categories listed above, output "Other." The output must be a single word only.

The types of actions are as follows:

- Pass
- Shoot
- Cross
- Dribble
- Successful Tackle
- Long Pass
- Header
- Other

Please output the corresponding action. The output must be a single word only.
===Example===
<output>
Shoot
===End of Example===

Let's begin!
<output>
"""

    return prompt

def get_response(img_path, prompt):
    response = ollama.chat(
        model = "llava",
        messages = [
            {
                "role": "user",
                "content": prompt,
                "images": [img_path]
            }
        ]
    )

    return response["message"]["content"]

def get_img_path_list(img_folder):
    img_path_list  = glob.glob(os.path.join(img_folder, '*'))
    img_path_list.sort()
    return img_path_list

def main():
    #* 対象ファイルのパス取得
    img_folder = "./data/output/720p/"
    img_path_list = get_img_path_list(img_folder)

    #* プロンプト設定
    prompt = action_analyzer_prompt()

    for i, img_path in enumerate(img_path_list):
        print(f"target path: {img_path}")
        response = get_response(img_path, prompt)
        print(f"Frame{i}: {response}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time  # 経過時間を計算
    print(f"main() 関数の処理時間: {elapsed_time:.6f} 秒")