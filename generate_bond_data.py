import random
from pathlib import Path
from typing import List, Dict, Tuple
import json

class BondQuoteDataGenerator:
    """债券询价数据生成器"""
    
    def __init__(self):
        # 字段定义
        self.side_values = [
            'bid', 'ofr', 'offer', 'buy', 'sell', 
            '收', '出', '买', '卖', '买入', '卖出'
        ]
        
        self.quantity_units = ['e', 'kw', 'w', '亿', '万', '']
        
        self.date_values = [
            't', 'tom', '今天', '明天', 
            '周一', '周二', '周三', '周四', '周五'
        ]
        
        self.speed_values = ['+0', '+1', '+2', '+3', '+4', '+5']
        
        # 噪声文本模板池（生产环境中用户可能夹杂的无关内容）
        self.noise_templates = [
            # 语气词 / 闲聊
            ['能做吗'], ['可以做吗'], ['能发吗'], ['能出吗'], ['能收吗'],
            ['有吗'], ['有的吗'], ['有券吗'], ['有没有'], ['还有吗'],
            ['麻烦了'], ['谢谢'], ['好的', '谢谢'], ['收到', '谢谢'],
            ['麻烦确认一下'], ['麻烦看下'], ['帮忙看一下'],
            ['麻烦报价'], ['麻烦给个价'], ['请问一下'],
            ['哥哥'], ['大佬'], ['哥'], ['兄弟'],
            # 拒绝 / 不做
            ['不做'], ['没兴趣'], ['算了'], ['不要了'], ['先不考虑'],
            ['太贵了'], ['价格不好'], ['收益率太低'], ['收益率太高'],
            # 催促 / 语气
            ['快一点'], ['急'], ['麻烦快点'], ['尽快'],
            ['回一个'], ['回复一下'], ['给个回复'],
            # 形容词 / 评价
            ['好券'], ['不错'], ['可以'], ['行'], ['没问题'],
            ['价很好'], ['价格不错'], ['价格可以'],
            ['收益率很高'], ['收益率偏低'], ['利率不错'],
            ['券不错'], ['这只券不错'], ['这个可以'],
            # 债券相关但非结构化描述（会作为噪声插入）
            ['25特长国债06'], ['24国开15'], ['23附息国债11'],
            ['30年国债'], ['10年国开'], ['5年政金债'],
            ['超长债'], ['长端利率债'], ['信用债'], ['城投债'],
            ['二级资本债'], ['永续债'], ['商金债'],
            ['新券'], ['老券'], ['活跃券'], ['次新券'],
            # 市场评论
            ['今天市场不错'], ['市场一般'], ['收益率上行'],
            ['收益率下行'], ['行情不好'], ['行情不错'],
            ['今天收益率上行比较多'], ['债市情绪不好'],
            ['资金面紧张'], ['资金面宽松'],
            ['央行今天操作'], ['资金偏紧'],
            # 交易相关
            ['具体聊'], ['私聊'], ['详聊'], ['聊一下'],
            ['加点微信'], ['加我微信'], ['电话沟通'],
            ['成交了吗'], ['成了吗'], ['成交了'],
            ['还有其他券吗'], ['有没有别的'],
            ['换一个'], ['换一只'],
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
        has_decimal = random.random() < 0.8  # 70%概率有小数
        
        if has_decimal:
            decimal_places = random.randint(1, 4)
            integer_part = random.randint(0, 5)
            decimal_part = ''.join([str(random.randint(0, 9)) for _ in range(decimal_places)])
            if random.random() < 0.85:
                return f"{integer_part}.{decimal_part}"
            else:
                return f"{integer_part}.{decimal_part}%"
        elif random.random() < 0.5:
            return str(random.randint(1, 5))
        else:
            return str(random.randint(1, 5)) + '%'
    
    def generate_quantity_value(self) -> str:
        """生成数量"""
        unit = random.choice(self.quantity_units)
        
        if unit in ['亿', 'e', 'kw']:
            # 亿级：0.1-99
            if random.random() < 0.5:
                # 整数
                value = random.randint(1, 99)
            else:
                # 小数
                decimal_places = random.randint(1, 2)
                decimal_part = ''.join([str(random.randint(0, 9)) for _ in range(decimal_places)])
                if decimal_part == '0' or decimal_part == '00':
                    integer_part = random.randint(1, 99)
                else:
                    integer_part = random.randint(0, 99)
                value = f"{integer_part}.{decimal_part}"
        else:
            # 万级：10-99999
            if random.random() < 0.5:
                # 整数
                value = random.randint(10, 99999)
            else:
                # 小数
                decimal_places = random.randint(1, 2)
                decimal_part = ''.join([str(random.randint(0, 9)) for _ in range(decimal_places)])
                integer_part = random.randint(10, 99999)
                value = f"{integer_part}.{decimal_part}"

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
        
        # 可选字段（80%概率包含）
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
        # 过滤掉特殊键
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
        
        # 生成元数据
        metadata = {
            'num_samples': len(sentences),
            'vocab_size': len(words),
            'char_vocab_size': len(chars),
            'tag_vocab_size': len(tags),
            'field_distribution': self.get_field_distribution()
        }
        
        with open(output_path / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def get_field_distribution(self, num_samples: int = 1000) -> Dict[str, float]:
        """获取字段分布统计"""
        stats = {'SIDE': 0, 'PRODUCT': 0, 'YIELD': 0, 
                'QUANTITY': 0, 'DATE': 0, 'SPEED': 0}
        
        for _ in range(num_samples):
            fields = self.generate_field_combination()
            for field in fields:
                if field in stats:
                    stats[field] += 1
        
        # 计算百分比
        for field in stats:
            stats[field] = stats[field] / num_samples * 100
        
        return stats


def main():
    """主函数"""
    print("🚀 开始生成债券询价数据集...")

    n_samples = 1000

    # 创建基础生成器
    generator = BondQuoteDataGenerator()
    
    # # 显示字段分布
    # print(f"\n📊 字段分布统计（采样{n_samples}条）：")
    # distribution = generator.get_field_distribution()
    # for field, percentage in distribution.items():
    #     print(f"  {field}: {percentage:.1f}%")
    
    # 生成数据集
    print("\n📝 生成数据集...")
    generator.generate_dataset(num_samples=n_samples, output_dir='./generated_data')
    
    print("\n✅ 数据生成完成！")


if __name__ == '__main__':
    main()
