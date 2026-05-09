import random
from pathlib import Path
from typing import List, Dict, Tuple, Union

DATASET_PATH = './generated_data'
SPACE_TOKEN = '<sp>'  # 特殊标记：表示原始文本中的空格

class BondQuoteDataGenerator:
    """债券询价数据生成器（字符级 IOBES 标注）"""

    def __init__(self):
        # 字段定义
        self.side_values = [
            'bid', 'ofr', 'offer', 'buy', 'sell',
            '收', '出', '买', '卖', '买入', '卖出'
        ]

        self.quantity_units = ['亿', 'e', 'k', 'kw', 'w', '万', 'm', 'mio', 'million', '']

        self.date_values = [
            't', 'tom', '今天', '明天', '今日', '明日', '今', '明',
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

        # 连写配置
        # DATE+SPEED 占 80%，其余 SIDE+QUANTITY / QUANTITY+SIDE / SIDE+PRODUCT 共占 20%
        self.concat_pair_weights = {
            ('DATE', 'SPEED'): 80,
            ('SIDE', 'QUANTITY'): 7,
            ('QUANTITY', 'SIDE'): 7,
            ('SIDE', 'PRODUCT'): 6,
        }

    def _char_iobes(self, text: str, tag_name: str) -> Tuple[List[str], List[str]]:
        """将文本转换为字符级 IOBES 标注，返回 (chars_list, tags_list)"""
        chars = list(text)
        n = len(chars)
        if n == 0:
            return [], []
        elif n == 1:
            return chars, [f'S-{tag_name}']
        else:
            tags = [f'B-{tag_name}'] + [f'I-{tag_name}'] * (n - 2) + [f'E-{tag_name}']
            return chars, tags

    def _inject_noise_chars(
        self, chars: List[str], tags: List[str], noise_ratio: float = 0.4
    ) -> Tuple[List[str], List[str]]:
        """
        在字符序列中随机插入噪声字符，噪声字符标签为 O。
        噪声模板中的多 token 条目，token 之间插入 <sp> 空格标记。
        """
        if random.random() > noise_ratio:
            return chars, tags

        num_noise = random.randint(1, 3)
        for _ in range(num_noise):
            noise_tokens = random.choice(self.noise_templates)
            noise_chars = []
            for j, token in enumerate(noise_tokens):
                if j > 0:
                    noise_chars.append(SPACE_TOKEN)
                noise_chars.extend(list(token))
            noise_tags = ['O'] * len(noise_chars)

            insert_pos = random.randint(0, len(chars))
            chars = chars[:insert_pos] + noise_chars + chars[insert_pos:]
            tags = tags[:insert_pos] + noise_tags + tags[insert_pos:]

        return chars, tags

    def _try_concatenate(
        self, fields: Dict[str, str]
    ) -> List[Union[Tuple[str, str], Tuple[str, List[Tuple[str, str]]]]]:
        """
        先对 fields 尝试连写组合，连写对作为整体返回，其余字段独立返回。
        
        返回列表，每个元素为 (value, tag_info)：
        - 非连写: tag_info 是 str（如 'SIDE'）
        - 连写:   tag_info 是 list（如 [('bid','SIDE'), ('2000w','QUANTITY')]）
        """
        field_names = list(fields.keys())

        # 根据权重随机选择一种连写对
        pairs, weights = zip(*self.concat_pair_weights.items())
        chosen_pair = random.choices(pairs, weights=weights, k=1)[0]

        # 找到匹配的连写对并合并
        used = set()
        result = []
        concat_applied = False

        # 优先尝试选中连写对
        if chosen_pair[0] in fields and chosen_pair[1] in fields:
            v1, v2 = fields[chosen_pair[0]], fields[chosen_pair[1]]
            result.append((v1 + v2, [(v1, chosen_pair[0]), (v2, chosen_pair[1])]))
            used.add(chosen_pair[0])
            used.add(chosen_pair[1])
            concat_applied = True

        # 剩余字段独立加入
        for name in field_names:
            if name not in used:
                result.append((fields[name], name))

        return result, concat_applied

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

        if rand < 0.5:
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
        """
        创建字符级 IOBES 标注的询价句子。
        
        流程：生成字段 → 先尝试连写组合 → 连写组合作为整体参与打乱顺序 → 字符级 IOBES 标注

        with_noise: 是否在句子中随机插入噪声内容（默认True）
        """
        # 先尝试连写（连写组合作为整体）
        field_entries, concat_applied = self._try_concatenate(fields)

        # 连写组合与非连写字段统一打乱顺序
        random.shuffle(field_entries)

        all_chars = []
        all_tags = []

        for i, (value, tag_info) in enumerate(field_entries):
            # 各字段/组合之间插入 <sp> 空格标记
            if i > 0:
                all_chars.append(SPACE_TOKEN)
                all_tags.append('O')

            if isinstance(tag_info, list):
                # 连写字段：分别对各原始字段值做字符级 IOBES
                for original_value, tag_name in tag_info:
                    chars, iobes_tags = self._char_iobes(original_value, tag_name)
                    all_chars.extend(chars)
                    all_tags.extend(iobes_tags)
            else:
                # 非连写字段
                chars, iobes_tags = self._char_iobes(value, tag_info)
                all_chars.extend(chars)
                all_tags.extend(iobes_tags)

        # 插入噪声
        if with_noise:
            all_chars, all_tags = self._inject_noise_chars(all_chars, all_tags)

        sentence = ' '.join(all_chars)
        tags_sentence = ' '.join(all_tags)
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
        """生成字符级词汇表"""
        # 字符集：每个句子是空格分隔的字符/标记，直接 split 即可
        chars = set()
        for sentence in sentences:
            chars.update(sentence.split())

        # 添加特殊标记
        chars.update(['<pad>', '<unk>', SPACE_TOKEN])

        with open(output_path / 'vocab.words.txt', 'w', encoding='utf-8') as f:
            for char in sorted(chars):
                f.write(char + '\n')

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

    n_samples = 50

    # 创建基础生成器
    generator = BondQuoteDataGenerator()

    # 生成数据集
    print("\n📝 生成数据集...")
    generator.generate_dataset(num_samples=n_samples, output_dir=DATASET_PATH)

    print("\n✅ 数据生成完成！")


if __name__ == '__main__':
    main()
