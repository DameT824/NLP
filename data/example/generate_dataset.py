import random
from pathlib import Path
from typing import List, Dict, Tuple

DATASET_PATH = './generated_data'

class BondQuoteDataGenerator:
    """债券询价数据生成器"""

    def __init__(self):
        # 字段定义
        self.side_values = [
            'bid', 'ofr', 'offer', 'buy', 'sell',
            '收', '出', '买', '卖', '买入', '卖出'
        ]

        self.quantity_units = ['e', 'k', 'kw', 'w', '亿', '万', '']

        self.date_values = [
            't', 'tom', '今天', '明天',
            '周一', '周二', '周三', '周四', '周五'
        ]

        self.speed_values = ['+0', '+1', '+2', '+3', '+4', '+5']

        # 噪声文本模板池（生产环境中用户可能夹杂的无关内容）
        self.noise_templates = [
            # 语气词 / 闲聊
            ['早'], ['需请示'], ['能做吗'], ['可以做吗'], ['能发吗'], ['能出吗'], ['能收吗'],
            ['有吗'], ['有的吗'], ['有券吗'], ['有没有'], ['还有吗'],
            ['麻烦确认一下'], ['麻烦看下'], ['帮忙看一下'],
            ['麻烦报价'], ['麻烦给个价'], ['请问一下'],
            ['@迟钧文-平安银行'], ['@张子健-平安银行'], ['@杨叶艺-平安银行'], ['给@付依孝-平安银行'], ['@李骅-平安银行'],
            # 债券相关但非结构化描述
            ['25特长国债06'], ['23附息国债11'], ['25附息国债12'], ['26附息国债07'], ['25超长特别国债06'], ['26超长特别国债02'], ['25南京银行科创债01BC'], ['25农发贴现07(免)'],
            ['25特2'], ['25t2'], ['24特6'], ['23t1'],
            ['24国开15'], ['25国开11'], ['26国开02'], ['25国开清发02'], ['22农发12'],
            ['23北京06'], ['25天津09'],
            ['22兴业银行CD313'], ['23浦发银行CD101'], ['22浦发银行CD154'], ['22北京银行CD097'], ['23中信银行CD161'], ['24浙商银行CD156'],
            ['114D', '23中国银行CD002'], ['91D', '23广发银行CD099'],
            ['30年国债'], ['10年国开'], ['5年政金债'], ['10Y国开'], ['5Y政金债'], ['10Y国债'],
            ['超长债'], ['长端利率债'], ['信用债'], ['城投债'], ['存单'], ['cd'], ['ncd'], ['农商'],
            ['二级资本债'], ['永续债'], ['商金债'], ['农发债'], ['政金债'], ['国开债'], ['政策性金融债'], ['其他金融机构债'],
            ['新券'], ['老券'], ['活跃券'], ['次新券'],
            ['97D'], ['316D'], ['308D'], ['345d'], ['136d'], ['41d'], ['255d'],
            ['9.9Y'], ['7.28Y'], ['7.62Y'], ['4.7956Y'], ['1.0192Y(休1)'], ['8.99Y(休1)'], ['3.9754Y(休1)'], ['9.7041Y(休2)'], ['28.99Y(休1)'],
            ['到期'], ['大量'], ['散量'], ['国际'], ['中债'], ['国利'], ['票面利率'], ['公募'], ['基金'],
            # 交易相关
            ['ref'], ['做'], ['tkn'], ['tks'], ['gvn'],
            ['成交了吗'], ['成了吗'], ['成交了'],
            ['其他'], ['还有其他券吗'], ['有没有别的'], ['具体聊'], ['私聊'], ['详聊'], ['聊一下'],
        ]

    def _inject_noise(self, words: List[str], tags: List[str], noise_ratio: float = 0.4) -> Tuple[List[str], List[str]]:
        """
        在已有的 tokens 序列中随机插入噪声词，噪声词标签为 O。

        noise_ratio: 插入噪声的概率（0-1），建议 0.3~0.5
        """
        if random.random() > noise_ratio:
            return words, tags

        # 随机选择 1~3 个噪声片段插入
        num_noise = random.randint(1, 3)

        for _ in range(num_noise):
            noise_tokens = random.choice(self.noise_templates)
            # 随机选择插入位置（0 到 len(words)）
            insert_pos = random.randint(0, len(words))

            for i, token in enumerate(noise_tokens):
                words.insert(insert_pos + i, token)
                tags.insert(insert_pos + i, 'O')

        return words, tags

    def generate_product_code(self) -> str:
        """生成券码"""
        digit_length = random.randint(6, 10)
        digits = ''.join([str(random.randint(0, 9)) for _ in range(digit_length)])

        # 50%概率添加.IB后缀
        if random.random() < 0.5:
            return f"{digits}.ib"
        return digits

    def generate_yield_value(self) -> str:
        """生成收益率"""
        has_decimal = random.random() < 0.8

        if has_decimal:
            decimal_places = random.randint(1, 4)
            decimal_part = random.randint(0, 10 ** decimal_places - 1)
            integer_part = random.randint(1, 4)
            if random.random() < 0.85:
                return f"{integer_part}.{decimal_part:0{decimal_places}d}"
            else:
                return f"{integer_part}.{decimal_part:0{decimal_places}d}%"
        elif random.random() < 0.3:
            return f"{random.randint(1, 5)}"
        else:
            return f"{random.randint(1, 5)}%"

    def generate_quantity_value(self) -> str:
        """生成数量"""
        unit = random.choice(self.quantity_units)

        if unit in ['亿', 'e', 'k', 'kw']:
            # 亿级：1-99
            if random.random() < 0.5:
                # 整数
                value = random.randint(1, 99)
            else:
                # 小数
                decimal_places = random.randint(1, 2)
                decimal_part = random.randint(0, 10 ** decimal_places - 1)
                integer_part = random.randint(1, 99)
                value = f"{integer_part}.{decimal_part:0{decimal_places}d}"
        else:
            # 万级：10-99999
            value = random.randint(10, 99999)

        return f"{value}{unit}"

    def generate_date_speed_fields(self) -> Dict[str, str]:
        rand = random.random()
        fields = {}

        if rand < 0.4:
            # 同时有 date 和 speed
            fields['DATE'] = random.choice(self.date_values)
            fields['SPEED'] = random.choice(self.speed_values)
        elif rand < 0.7:
            # 只有 date
            fields['DATE'] = random.choice(self.date_values)
        elif rand < 0.9:
            # 只有 speed
            fields['SPEED'] = random.choice(self.speed_values)
        # 0.9~1.0: 两者都不有

        return fields

    def generate_field_combination(self) -> Dict[str, str]:
        """生成字段组合"""
        # 必填字段
        side = random.choice(self.side_values)
        product = self.generate_product_code()

        # 可选字段
        fields = {
            'SIDE': side,
            'PRODUCT': product
        }

        if random.random() < 0.8:
            fields['YIELD'] = self.generate_yield_value()

        if random.random() < 0.8:
            fields['QUANTITY'] = self.generate_quantity_value()

        if random.random() < 0.7:
            ds_fields = self.generate_date_speed_fields()
            fields.update(ds_fields)

        return fields

    def create_quote_sentence(self, fields: Dict[str, str], with_noise: bool = True) -> Tuple[str, str]:
        """创建询价句子和标注

        with_noise: 是否在句子中随机插入噪声内容（默认True）
        """
        field_order = [k for k in fields.keys()]
        random.shuffle(field_order)

        words = []
        tags = []

        for field_name in field_order:
            field_value = fields[field_name]
            field_words = field_value.split()

            # 处理多token字段
            if len(field_words) == 1:
                tags.append(f"S-{field_name}")
                words.extend(field_words)
            elif len(field_words) > 1:
                tags.append(f"B-{field_name}")
                for _ in range(len(field_words) - 2):
                    tags.append(f"I-{field_name}")
                tags.append(f"E-{field_name}")
                words.extend(field_words)

        # 插入噪声内容
        if with_noise:
            words, tags = self._inject_noise(words, tags)

        sentence = ' '.join(words)
        tags_sentence = ' '.join(tags)

        return sentence, tags_sentence

    def generate_single_sample(self, with_noise: bool = True) -> Tuple[str, str]:
        """生成单个样本"""
        fields = self.generate_field_combination()
        return self.create_quote_sentence(fields, with_noise=with_noise)

    def generate_dataset(self, num_samples: int, output_dir: str = './generated_data'):
        """生成完整数据集

        训练集中 60% 含噪声样本 + 40% 纯净样本，验证/测试集保持相同比例
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        noise_ratio = 0.6  # 训练集中含噪声样本的比例

        # 生成训练数据
        train_words = []
        train_tags = []
        num_noisy = int(num_samples * noise_ratio)
        for i in range(num_samples):
            with_noise = i < num_noisy
            sentence, tags = self.generate_single_sample(with_noise=with_noise)
            train_words.append(sentence)
            train_tags.append(tags)

        # 保存训练数据
        with open(output_path / 'train.words.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_words))

        with open(output_path / 'train.tags.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_tags))

        # 生成验证数据（同样比例）
        val_words = []
        val_tags = []
        val_noisy = int(num_samples * 0.1 * noise_ratio)
        for i in range(int(num_samples * 0.1)):
            with_noise = i < val_noisy
            sentence, tags = self.generate_single_sample(with_noise=with_noise)
            val_words.append(sentence)
            val_tags.append(tags)

        with open(output_path / 'val.words.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(val_words))

        with open(output_path / 'val.tags.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(val_tags))

        # 生成测试数据
        test_words = []
        test_tags = []
        test_noisy = int(num_samples * 0.1 * noise_ratio)
        for i in range(int(num_samples * 0.1)):
            with_noise = i < test_noisy
            sentence, tags = self.generate_single_sample(with_noise=with_noise)
            test_words.append(sentence)
            test_tags.append(tags)

        with open(output_path / 'test.words.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_words))

        with open(output_path / 'test.tags.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_tags))

        # 生成词汇表
        self.generate_vocab(train_words, output_path)

        print(f"✅ 数据生成完成！")
        print(f"训练集: {len(train_words)} 条（含噪声: {num_noisy}, 纯净: {num_samples - num_noisy}）")
        print(f"验证集: {len(val_words)} 条")
        print(f"测试集: {len(test_words)} 条")
        print(f"输出目录: {output_path}")

    def generate_vocab(self, sentences: List[str], output_path: Path):
        """生成词汇表"""
        # 词表
        words = set()
        for sentence in sentences:
            words.update(sentence.split())

        # 添加特殊标记
        words.update(['<pad>', '<unk>'])

        with open(output_path / 'vocab.words.txt', 'w', encoding='utf-8') as f:
            for word in sorted(words):
                f.write(word + '\n')

        # 字符表
        chars = set()
        for sentence in sentences:
            for word in sentence.split():
                chars.update(list(word))

        chars.update(['<pad>', '<unk>'])

        with open(output_path / 'vocab.chars.txt', 'w', encoding='utf-8') as f:
            for char in sorted(chars):
                f.write(char + '\n')

        # 标签表
        tags = ['O']
        for field in ['SIDE', 'PRODUCT', 'YIELD', 'QUANTITY', 'DATE', 'SPEED']:
            tags.extend([f'B-{field}', f'I-{field}', f'E-{field}', f'S-{field}'])

        with open(output_path / 'vocab.tags.txt', 'w', encoding='utf-8') as f:
            for tag in tags:
                f.write(tag + '\n')


def main():
    """主函数"""
    print("🚀 开始生成债券询价数据集...")

    n_samples = 50000

    # 创建基础生成器
    generator = BondQuoteDataGenerator()

    # 生成数据集
    print("\n📝 生成数据集...")
    generator.generate_dataset(num_samples=n_samples, output_dir=DATASET_PATH)

    print("\n✅ 数据生成完成！")


if __name__ == '__main__':
    main()
