import torch
from transformers import BertTokenizerFast
from train_bert_ner import BertCRFNER
from typing import List
from dataclasses import dataclass
from train_bert_ner import MODEL_PATH, WEIGHTS_PATH

# ============================================================
# 配置
# ============================================================

MAX_LENGTH = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    'SPEED': '速度',
}


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Entity:
    """识别出的实体"""
    type: str  # 实体类型：SIDE, PRODUCT, YIELD, QUANTITY, DATE, SPEED
    text: str  # 实体原文
    start: int  # 实体在原文中的起始字符位置（包含）
    end: int  # 实体在原文中的结束字符位置（不包含）
    description: str = ''  # 实体中文描述


@dataclass
class NERResult:
    """NER 识别结果"""
    text: str  # 原始文本
    tokens: List[str]  # 分词结果
    tags: List[str]  # 每个 token 的标签
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

    def __init__(self, model_dir: str = MODEL_PATH, weights_path: str = WEIGHTS_PATH):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        self.model = BertCRFNER(len(TAG2IDX), bert_model_name=model_dir)
        self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        self.model.to(DEVICE)
        self.model.eval()

    @staticmethod
    def _is_chinese(ch: str) -> bool:
        return '\u4e00' <= ch <= '\u9fff'

    def _split_date_speed(self, text: str) -> List[str]:
        """
        预处理语料，将DATE+SPEED连在一起的字符串分隔开
        仅在以下情况执行分隔：
        1. part中包含'+'
        2. '+'前面有连续字符（即DATE和SPEED连在一起）

        支持的模式：
        - t+0, t+1, t+2, t+3, t+4, t+5
        - tom+0, tom+1, ...
        - 今天+0, 今日+1, 明天+2, 明日+3, ...
        - 周一+1, 周二+2, 周三+3, 周四+4, 周五+5

        例如:
        - "t+0" -> ["t", "+0"]
        - "tom+1" -> ["tom", "+1"]
        - "今天+2" -> ["今天", "+2"]
        - "t +0" -> ["t +0"] (已用空格分隔，无需处理)
        """
        # 检查是否包含'+'
        if '+' not in text:
            return [text]

        # 找到'+'的位置
        plus_index = text.index('+')

        # 检查'+'前面是否有连续字符（非空格）
        if plus_index == 0:
            # '+'在开头，如"+0"，无需分隔
            return [text]

        # 检查'+'前面是否是空格（如"t +0"）
        if text[plus_index - 1] == ' ':
            # 已经用空格分隔，无需处理
            return [text]

        # DATE值列表（按长度降序排列，优先匹配长字符串）
        date_values = [
            '今天', '今日', '明天', '明日',
            '周一', '周二', '周三', '周四', '周五',
            'tom', 't'
        ]

        # SPEED值列表
        speed_values = ['+0', '+1', '+2', '+3', '+4', '+5']

        # 提取'+'前面的部分（DATE）和'+'后面的部分（SPEED）
        before_plus = text[:plus_index]
        after_plus = text[plus_index:]

        # 尝试匹配DATE值
        matched_date = None
        for date_val in date_values:
            if before_plus.endswith(date_val):
                matched_date = date_val
                break

        if matched_date is None:
            # 没有匹配到DATE值，返回原字符串
            return [text]

        # 尝试匹配SPEED值
        matched_speed = None
        for speed_val in speed_values:
            if after_plus.startswith(speed_val):
                matched_speed = speed_val
                break

        if matched_speed is None:
            # 没有匹配到SPEED值，返回原字符串
            return [text]

        # 成功匹配DATE和SPEED，进行分隔
        # 处理DATE前面的部分（如果有）
        tokens = []
        prefix = before_plus[:-len(matched_date)]
        if prefix:
            tokens.append(prefix)

        tokens.append(matched_date)
        tokens.append(matched_speed)

        # 处理SPEED后面的部分（如果有）
        suffix = after_plus[len(matched_speed):]
        if suffix:
            tokens.append(suffix)

        return tokens

    def _tokenize_text(self, text: str) -> tuple:
        """
        对输入文本进行分词，返回 (tokens, char_offsets)

        处理流程：
        1. 按空格分割文本
        2. 对每个part，如果包含'+'且前面有连续字符，则进行DATE+SPEED分隔
        3. 记录每个token在原文中的字符偏移

        char_offsets: 每个token在原文中的(start, end)字符偏移
        """
        tokens = []
        char_offsets = []

        # 按空格分割
        parts = text.split()

        for part in parts:
            if not part:
                continue

            # 找到该part在原文中的起始位置
            part_start = text.find(part)

            # 仅当part包含'+'时才进行DATE+SPEED分隔处理
            if '+' in part:
                sub_tokens = self._split_date_speed(part)
            else:
                sub_tokens = [part]

            # 计算每个sub_token的字符偏移
            offset = part_start
            for sub_token in sub_tokens:
                tokens.append(sub_token)
                char_offsets.append((offset, offset + len(sub_token)))
                offset += len(sub_token)

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
