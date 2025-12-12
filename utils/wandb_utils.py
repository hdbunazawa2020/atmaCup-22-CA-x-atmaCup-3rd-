import os
import wandb
def set_wandb(config):
    # set environment variables
    os.environ["WANDB_BASE_URL"] = "https://toyota.wandb.io" # トヨタのwandb
    os.environ["WANDB_PROJECT"] = config["project"]

    # get wandb_api_key (for TOYOTA in house environment)
    wandb_api_key = os.getenv("WANDB_API_KEY") # export wandb_api_key before running program
    if not wandb_api_key:
        raise EnvironmentError("WANDB_API_KEY is not set in the environment. Please set it before running the script.")
    os.environ["WANDB_API_KEY"] = wandb_api_key

    # その他の設定
    os.environ["NCCL_DEBUG"] = "INFO"  # NCCLのデバッグ情報を出力
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 使用するGPUを指定
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"  # トークナイザーの並列実行を無効化

    # WandBの初期化（必要であれば）
    wandb.login()

from tqdm import tqdm
def set_wandb_table(oof_df, columns):
    # ✨ W&B: Create a Table to store predictions for each test step
    table = wandb.Table(columns=columns)
    obj_cols = oof_df.select_dtypes(include=['object']).columns.tolist()

    # tqdmで進捗バーを表示しながらデータを処理
    for i, (index, row) in enumerate(tqdm(oof_df.iterrows(), total=len(oof_df))):
        row_data = []
        for col in columns:
            if col in obj_cols:
                row_data.append(str(row[col]))
            else:
                row_data.append(row[col])
        table.add_data(*row_data) 
    # Log the table to W&B
    wandb.log({"oof_table": table})