# ner_service.py
"""债券询价语料要素识别服务"""

import torch
from transformers import BertTokenizerFast
from train_bert_ner import BertCRFNER
from typing import List
from dataclasses import dataclass

# ============================================================
# 配置
# ============================================================

MODEL_DIR = './models/bert-base-chinese'
WEIGHTS_PATH = 'bert_crf_bond_ner.pth'
MAX_LENGTH = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

# 标签映射（与训练时保持一致）
TAG2IDX = {'O': 0, '<PAD>': 1}
for field in ['SIDE', 'PRODUCT', 'YIELD', 'QUANTITY', 'DATE', 'SPEED']:
    for prefix in ['B', 'I', 'E', 'S']:
        TAG2IDX[f'{prefix}-{field}'] = len(TAG2IDX)

IDX2TAG = {v: k for k, v in TAG2IDX.items()}

# BMES 标签含义
TAG_DESCRIPTION = {
    'SIDE': '方向（买/卖/OFR/BID）',
    'PRODUCT': '债券代码/名称',
    'YIELD': '收益率',
    'QUANTITY': '数量/金额',
    'DATE': '日期/期限',
    'SPEED': '利差（BP）',
}


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Entity:
    """识别出的实体"""
    type: str           # 实体类型：SIDE, PRODUCT, YIELD, QUANTITY, DATE, SPEED
    text: str           # 实体原文
    start: int          # 实体在原文中的起始字符位置（包含）
    end: int            # 实体在原文中的结束字符位置（不包含）
    description: str = ''  # 实体中文描述


@dataclass
class NERResult:
    """NER 识别结果"""
    text: str           # 原始文本
    tokens: List[str]   # 分词结果
    tags: List[str]     # 每个 token 的标签
    entities: List[Entity]  # 识别出的实体列表

    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'tokens': self.tokens,
            'tags': self.tags,
            'entities': [
                {
                    'type': e.type,
                    'text': e.text,
                    'start': e.start,
                    'end': e.end,
                    'description': e.description,
                }
                for e in self.entities
            ]
        }

    def __repr__(self) -> str:
        lines = [f"原文: {self.text}"]
        if self.entities:
            lines.append("识别结果:")
            for e in self.entities:
                lines.append(f"  [{e.type}] {e.text} ({e.description})")
        else:
            lines.append("未识别到任何实体")
        return '\n'.join(lines)


# ============================================================
# 核心服务
# ============================================================

class BondQuoteNERService:
    """债券询价 NER 服务"""

    def __init__(self, model_dir: str = MODEL_DIR, weights_path: str = WEIGHTS_PATH):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        self.model = BertCRFNER(len(TAG2IDX), bert_model_name=model_dir)
        self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        self.model.to(DEVICE)
        self.model.eval()

    @staticmethod
    def _is_chinese(ch: str) -> bool:
        return '\u4e00' <= ch <= '\u9fff'

    @staticmethod
    def _split_token_by_boundaries(token: str) -> List[str]:
        """
        按边界分割 token（中文/数字/字母/符号边界）
        例如: "出2000w1.985" -> ["出", "2000w", "1.985"]
              "250203.ib"   -> ["250203", ".ib"]
        """
        if not token:
            return []

        sub_tokens = []
        start = 0

        for i in range(1, len(token)):
            curr, prev = token[i], token[i - 1]

            is_curr_cjk = BondQuoteNERService._is_chinese(curr)
            is_prev_cjk = BondQuoteNERService._is_chinese(prev)
            is_curr_digit = curr.isdigit()
            is_prev_digit = prev.isdigit()
            is_curr_letter = curr.isalpha()
            is_prev_letter = prev.isalpha()

            boundary = False
            if is_curr_cjk != is_prev_cjk:
                # 中文 <-> 非中文（含英文标点等）
                boundary = True
            elif (is_curr_digit and is_prev_letter) or (is_curr_letter and is_prev_digit):
                # 数字 <-> 字母（如 "2000w", "ib250203"）
                boundary = True
            elif (is_curr_digit or is_curr_letter) and not (is_prev_digit or is_prev_letter) and not is_prev_cjk:
                # 数字/字母 <-> 符号（如 "1.985", ".ib"）
                boundary = True
            elif not (is_curr_digit or is_curr_letter or is_curr_cjk) and (is_prev_digit or is_prev_letter) and not is_prev_cjk:
                # 符号 <-> 数字/字母（如 "250203.", "w1"）
                boundary = True

            if boundary:
                sub_tokens.append(token[start:i])
                start = i

        if start < len(token):
            sub_tokens.append(token[start:])

        return sub_tokens

    def _tokenize_text(self, text: str) -> tuple:
        """
        对输入文本进行分词，返回 (tokens, char_offsets)
        支持空格分割 + 数字/字母/中文边界智能分割，
        确保无空格输入（如 "出2000w1.985250203.ib"）也能正确分词。
        char_offsets: 每个 token 在原文中的 (start, end) 字符偏移
        """
        tokens = []
        char_offsets = []
        i = 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            # 找到连续非空格字符作为一个 chunk
            chunk_start = i
            while i < len(text) and not text[i].isspace():
                i += 1
            chunk = text[chunk_start:i]

            # 按 中文/数字/字母/符号 边界进一步分割
            sub_tokens = self._split_token_by_boundaries(chunk)
            if sub_tokens:
                # 计算每个 sub_token 在原文中的偏移
                offset = chunk_start
                for sub in sub_tokens:
                    tokens.append(sub)
                    char_offsets.append((offset, offset + len(sub)))
                    offset += len(sub)

        return tokens, char_offsets

    def _extract_entities(self, tokens: List[str], tags: List[str],
                          char_offsets: list) -> List[Entity]:
        """从 BMES 标签序列中提取实体"""
        entities = []
        current_type = None
        current_tokens = []
        current_start = None
        current_end = None

        for idx, (token, tag) in enumerate(zip(tokens, tags)):
            if tag.startswith('B-') or tag.startswith('S-'):
                # 先保存上一个未关闭的实体
                if current_type:
                    entities.append(Entity(
                        type=current_type,
                        text=''.join(current_tokens),
                        start=current_start,
                        end=current_end,
                        description=TAG_DESCRIPTION.get(current_type, '')
                    ))
                etype = tag[2:]
                current_type = etype
                current_tokens = [token]
                current_start, current_end = char_offsets[idx]

                if tag.startswith('S-'):
                    # 单字实体，直接保存
                    entities.append(Entity(
                        type=etype,
                        text=token,
                        start=current_start,
                        end=current_end,
                        description=TAG_DESCRIPTION.get(etype, '')
                    ))
                    current_type = None
                    current_tokens = []

            elif tag.startswith('I-') and current_type:
                etype = tag[2:]
                if etype == current_type:
                    current_tokens.append(token)
                    current_end = char_offsets[idx][1]

            elif tag.startswith('E-') and current_type:
                etype = tag[2:]
                if etype == current_type:
                    current_tokens.append(token)
                    current_end = char_offsets[idx][1]
                    entities.append(Entity(
                        type=current_type,
                        text=''.join(current_tokens),
                        start=current_start,
                        end=current_end,
                        description=TAG_DESCRIPTION.get(current_type, '')
                    ))
                    current_type = None
                    current_tokens = []

            else:
                # O 标签或类型不匹配，关闭当前实体
                if current_type:
                    entities.append(Entity(
                        type=current_type,
                        text=''.join(current_tokens),
                        start=current_start,
                        end=current_end,
                        description=TAG_DESCRIPTION.get(current_type, '')
                    ))
                    current_type = None
                    current_tokens = []

        # 处理末尾未关闭的实体
        if current_type:
            entities.append(Entity(
                type=current_type,
                text=''.join(current_tokens),
                start=current_start,
                end=current_end,
                description=TAG_DESCRIPTION.get(current_type, '')
            ))

        return entities

    def predict(self, text: str) -> NERResult:
        """
        对单条文本进行 NER 识别

        Args:
            text: 输入的债券询价语料

        Returns:
            NERResult 识别结果
        """
        # 分词
        tokens, char_offsets = self._tokenize_text(text)
        if not tokens:
            return NERResult(text=text, tokens=[], tags=[], entities=[])

        # Tokenize
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)

        # 预测
        with torch.no_grad():
            predictions = self.model(input_ids, attention_mask)

        # 将 sub-token 标签对齐回 word 级别
        word_ids = encoding.word_ids()
        pred_labels = predictions[0]
        word_tags = ['O'] * len(tokens)

        for pos, word_idx in enumerate(word_ids):
            if word_idx is not None:
                word_tags[word_idx] = IDX2TAG[pred_labels[pos]]

        # 提取实体
        entities = self._extract_entities(tokens, word_tags, char_offsets)

        return NERResult(
            text=text,
            tokens=tokens,
            tags=word_tags,
            entities=entities,
        )

    def predict_batch(self, texts: List[str]) -> List[NERResult]:
        """批量预测"""
        return [self.predict(text) for text in texts]


# ============================================================
# 本地开发自测入口
# ============================================================

if __name__ == '__main__':
    print("正在加载模型...")
    service = BondQuoteNERService()
    print(f"模型加载完成, 设备: {DEVICE}")
    print("输入语料进行解析，输入 q 退出\n" + "-" * 50)

    while True:
        try:
            text = input("\n请输入语料: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break
        if text in ('q', 'quit', 'exit'):
            break
        if not text:
            continue
        result = service.predict(text)
        print(result)
        print(f"\nJSON: {result.to_dict()}")
