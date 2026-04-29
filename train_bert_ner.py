import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertForTokenClassification
from TorchCRF import CRF
from pathlib import Path
import time
import shutil
from tqdm import tqdm
from generate_bond_data import DATASET_PATH

MODEL_PATH = './models/bert-base-chinese'
OUTPUT_MODEL_PATH = './models/bert_output'
WEIGHTS_FILE = 'bert_crf_bond_ner.pth'
WEIGHTS_PATH = f'{OUTPUT_MODEL_PATH}/{WEIGHTS_FILE}'


class BondQuoteDataset(Dataset):
    """债券询价数据集"""

    def __init__(self, words_file, tags_file, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 读取数据
        with open(words_file, 'r', encoding='utf-8') as f:
            self.words = [line.strip().split() for line in f]

        with open(tags_file, 'r', encoding='utf-8') as f:
            self.tags = [line.strip().split() for line in f]

        # 标签映射
        self.tag2idx = {'O': 0, '<PAD>': 1}
        for field in ['SIDE', 'PRODUCT', 'YIELD', 'QUANTITY', 'DATE', 'SPEED']:
            self.tag2idx.update({
                f'B-{field}': len(self.tag2idx),
                f'I-{field}': len(self.tag2idx),
                f'E-{field}': len(self.tag2idx),
                f'S-{field}': len(self.tag2idx)
            })
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        words = self.words[idx]
        tags = self.tags[idx]

        # Tokenize
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 对齐标签
        labels = []
        word_ids = encoding.word_ids()
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(1)  # <PAD>
            else:
                labels.append(self.tag2idx[tags[word_idx]])

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


class BertCRFNER(torch.nn.Module):
    """BERT + CRF 模型"""

    def __init__(self, num_labels, bert_model_name='bert-base-chinese'):
        super().__init__()
        self.bert = BertForTokenClassification.from_pretrained(
            bert_model_name,
            num_labels=num_labels
        )
        self.crf = CRF(num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        if labels is not None:
            # 训练模式
            crf_loss = self.crf(logits, labels, mask=attention_mask.bool())
            if crf_loss.dim() > 0:
                crf_loss = crf_loss.mean()
            return -crf_loss
        else:
            # 预测模式
            predictions = self.crf.viterbi_decode(logits, mask=attention_mask.bool())
            return predictions


def train_model():
    """训练模型"""
    # 配置
    config = {
        'batch_size': 16,
        'learning_rate': 2e-5,
        'num_epochs': 5,
        'max_length': 64,
        'bert_model': MODEL_PATH
    }

    # 加载tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(config['bert_model'])

    # 创建数据集
    train_dataset = BondQuoteDataset(
        f'{DATASET_PATH}/train.words.txt',
        f'{DATASET_PATH}/train.tags.txt',
        tokenizer,
        config['max_length']
    )

    val_dataset = BondQuoteDataset(
        f'{DATASET_PATH}/val.words.txt',
        f'{DATASET_PATH}/val.tags.txt',
        tokenizer,
        config['max_length']
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # 创建模型
    num_labels = len(train_dataset.tag2idx)
    model = BertCRFNER(num_labels, config['bert_model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # 指定使用GPU 0
    model.to(device)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])

    # 训练循环
    print(f"\n{'=' * 50}")
    print(f"设备: {device}")
    print(f"训练集: {len(train_dataset)} 条, 验证集: {len(val_dataset)} 条")
    print(f"每epoch批次数: {len(train_loader)}")
    print(f"{'=' * 50}\n")

    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}")
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{total_loss / (batch_idx + 1):.4f}'})

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1}/{config['num_epochs']} 完成, Loss: {avg_loss:.4f}, 耗时: {epoch_time:.1f}s")

        # 验证
        print("正在验证...")
        val_accuracy = evaluate_model(model, val_loader, device)
        print(f"Validation Accuracy: {val_accuracy:.4f}\n")

    output_dir = Path(OUTPUT_MODEL_PATH)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"输出目录已新建: {output_dir}")

    # 保存模型
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print("✅ 模型训练完成并保存！")


def evaluate_model(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            predictions = model(input_ids, attention_mask)

            # 计算准确率
            mask = attention_mask.bool()
            for i in range(len(predictions)):
                preds = torch.tensor(predictions[i][:mask[i].sum()], device=device)
                labs = labels[i][:mask[i].sum()]
                total_correct += (preds == labs).sum().item()
                total_samples += len(preds)

    return total_correct / total_samples


if __name__ == '__main__':
    train_model()
