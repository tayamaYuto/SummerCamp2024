import ollama
import glob
import os

def action_analyzer_prompt():
    prompt = """
あなたはサッカーの試合映像からプレイヤーのアクションを分析する能力を持っています。
サッカーの試合映像を見て、プレイヤーのアクションが以下のリストのどのアクションに該当するかを分析してください。
もし、アクションが上記のいずれにも該当しない場合は、「その他」と出力してください。
出力は単語のみです。

アクションの種類は以下の通りです：

1. パス
2. シュート
3. クロス
4. ドリブル
5. 成功したタックル
6. ロングパス
7. ヘディング
8. その他

該当するアクションを出力してください。出力形式の例は以下を参照してください。
===例===
<output>
シュート
===例の終わり===

それでは始めましょう！
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
    main()