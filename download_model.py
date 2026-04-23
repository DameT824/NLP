from modelscope import AutoTokenizer, AutoModelForCausalLM, snapshot_download
from transformers import BertTokenizer, BertModel
from pathlib import Path

MODEL_DIR='./modelsssxxxx'

# CLI Download
# export MODELSCOPE_CACHE="xxxxx"
# modelscope download --model google-bert/bert-base-chinese --local_dir ./models/bert-base-chinese


# snapshot download
def snapshot_download_model(model_name):
    local_path = Path(MODEL_DIR)
    local_path.mkdir(parents=True, exist_ok=True)

    print(f"正在下载模型到本地: {local_path}")
    print("这可能需要几分钟，请耐心等待...")

    model_dir = snapshot_download(model_name, cache_dir=str(local_path))
    print("✓ 模型下载完成")
    print(f"\n✅ 模型已保存到: {local_path}")
    print(f"占用空间: {sum(f.stat().st_size for f in local_path.rglob('*') if f.is_file()) / (1024 ** 3):.2f} GB")
    return model_dir


def download_model(model_name):
    local_path = Path(MODEL_DIR)
    local_path.mkdir(parents=True, exist_ok=True)

    print(f"正在下载模型到本地: {local_path}")
    print("这可能需要几分钟，请耐心等待...")

    # 下载 tokenizer
    print("\n[1/2] 下载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(local_path))
    tokenizer.save_pretrained(str(local_path))
    print("✓ Tokenizer 下载完成")

    # 下载模型
    print("\n[2/2] 下载 BERT 模型...")
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=str(local_path))
    model.save_pretrained(str(local_path))
    print("✓ 模型下载完成")
    print(f"\n✅ 模型已保存到: {local_path}")
    print(f"占用空间: {sum(f.stat().st_size for f in local_path.rglob('*') if f.is_file()) / (1024 ** 3):.2f} GB")
    return str(Path(MODEL_DIR))


if __name__ == "__main__":
    # model_name can be model name (eg: 'google-bert/bert-base-chinese', which will download online)
    # or local model path (eg: './models/bert-base-chinese', which will load locally)
    # model_name = snapshot_download_model('google-bert/bert-base-chinese')
    model_name = download_model('google-bert/bert-base-chinese')

    # Use transformers lib
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Use modelscope lib
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name)

