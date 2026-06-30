import os
from huggingface_hub import snapshot_download

# 使用国内镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def download_to_d_drive():
    # 明确指定 D 盘路径
    target_dir = r"D:\WZU_Models"

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建目录: {target_dir}")

    print(f"正在下载模型至 {target_dir}，请保持网络连接...")
    snapshot_download(
        repo_id="facebook/mbart-large-50-many-to-many-mmt",
        local_dir=r"D:\WZU_Models",
        local_dir_use_symlinks=False,
        # 核心：忽略掉你不需要的格式（TensorFlow, Flax 等）
        ignore_patterns=["*.h5", "*.msgpack", "*.ot", "*.msgpack"]
    )

    # 顺便初始化项目内的 data 文件夹用于存放词典
    if not os.path.exists("./data"):
        os.makedirs("./data")
        with open("./data/dict.json", "w", encoding="utf-8") as f:
            f.write('{"Transformer": "变换器模型", "Wenzhou University": "温州大学"}')

    print(f"\n下载成功！模型已存入: {target_dir}")


if __name__ == "__main__":
    download_to_d_drive()