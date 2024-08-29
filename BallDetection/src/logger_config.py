import logging

# ロガーの取得
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # ログレベルの設定

# 既にハンドラーが設定されているか確認
if not logger.hasHandlers():
    # コンソールへの出力設定
    console_handler = logging.StreamHandler()

    # ログフォーマットの設定
    formatter = logging.Formatter('%(filename)s - %(lineno)d - %(funcName)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # ハンドラーをロガーに追加
    logger.addHandler(console_handler)