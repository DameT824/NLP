import random
import itertools
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
            '周一', '周二', '周三', '周四', '周五', ''
        ]
        
        self.speed_values = ['+0', '+1', '+2', '']
        
    def generate_product_code(self) -> str:
        """生成券码"""
        digit_length = random.randint(6, 10)
        digits = ''.join([str(random.randint(0, 9)) for _ in range(digit_length)])
        
        # 50%概率添加.IB后缀
        if random.random() < 0.5:
            return f"{digits}.IB"
        return digits
    
    def generate_yield_value(self) -> str:
        """生成收益率"""
        has_decimal = random.random() < 0.7  # 70%概率有小数
        
        if has_decimal:
            decimal_places = random.randint(1, 4)
            integer_part = random.randint(0, 10)
            decimal_part = ''.join([str(random.randint(0, 9)) for _ in range(decimal_places)])
            return f"{integer_part}.{decimal_part}"
        else:
            return str(random.randint(0, 15))
    
    def generate_quantity_value(self) -> str:
        """生成数量"""
        unit = random.choice(self.quantity_units)
        
        if unit in ['亿', 'e']:
            # 亿级：0.1-10
            if random.random() < 0.5:
                # 整数
                value = random.randint(1, 10)
            else:
                # 小数
                decimal_places = random.randint(1, 2)
                integer_part = random.randint(0, 9)
                decimal_part = ''.join([str(random.randint(0, 9)) for _ in range(decimal_places)])
                value = f"{integer_part}.{decimal_part}"
        elif unit in ['万', 'w', 'kw', '']:
            # 万级：100-10000
            if random.random() < 0.3:
                # 简化表示：5000w
                value = str(random.randint(100, 10000))
            else:
                # 完整表示：5000
                value = str(random.randint(1000, 100000))
        
        if unit:
            return f"{value}{unit}"
        return value
    
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
        
        if random.random() < 0.6:
            fields['DATE'] = random.choice(self.date_values)
        
        if random.random() < 0.6:
            fields['SPEED'] = random.choice(self.speed_values)
        
        return fields
    
    def create_quote_sentence(self, fields: Dict[str, str]) -> Tuple[str, List[Tuple[str, str]]]:
        """创建询价句子和标注"""
        field_order = list(fields.keys())
        random.shuffle(field_order)
        
        words = []
        tags = []
        
        for field_name in field_order:
            field_value = fields[field_name]
            field_words = field_value.split()
            
            # 处理多token字段
            if len(field_words) == 1:
                tags.append(f"S-{field_name}")
            else:
                tags.append(f"B-{field_name}")
                for _ in range(len(field_words) - 2):
                    tags.append(f"I-{field_name}")
                tags.append(f"E-{field_name}")
            
            words.extend(field_words)
        
        sentence = ' '.join(words)
        tags_sentence = ' '.join(tags)
        
        return sentence, tags_sentence
    
    def generate_single_sample(self) -> Tuple[str, str]:
        """生成单个样本"""
        fields = self.generate_field_combination()
        return self.create_quote_sentence(fields)
    
    def generate_dataset(self, num_samples: int, output_dir: str = './generated_data'):
        """生成完整数据集"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 生成训练数据
        train_words = []
        train_tags = []
        for _ in range(num_samples):
            sentence, tags = self.generate_single_sample()
            train_words.append(sentence)
            train_tags.append(tags)
        
        # 保存训练数据
        with open(output_path / 'train.words.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_words))
        
        with open(output_path / 'train.tags.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_tags))
        
        # 生成验证数据
        val_words = []
        val_tags = []
        for _ in range(int(num_samples * 0.1)):
            sentence, tags = self.generate_single_sample()
            val_words.append(sentence)
            val_tags.append(tags)
        
        with open(output_path / 'val.words.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(val_words))
        
        with open(output_path / 'val.tags.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(val_tags))
        
        # 生成测试数据
        test_words = []
        test_tags = []
        for _ in range(int(num_samples * 0.1)):
            sentence, tags = self.generate_single_sample()
            test_words.append(sentence)
            test_tags.append(tags)
        
        with open(output_path / 'test.words.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_words))
        
        with open(output_path / 'test.tags.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_tags))
        
        # 生成词汇表
        self.generate_vocab(train_words, output_path)
        
        print(f"✅ 数据生成完成！")
        print(f"训练集: {len(train_words)} 条")
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
                stats[field] += 1
        
        # 计算百分比
        for field in stats:
            stats[field] = stats[field] / num_samples * 100
        
        return stats


class AdvancedDataGenerator(BondQuoteDataGenerator):
    """高级数据生成器，包含更多真实场景模拟"""
    
    def __init__(self):
        super().__init__()
        
        # 添加口语化表达
        self.side_variations = {
            'bid': ['bid', '买价', '买入价', '求购'],
            'ofr': ['ofr', 'offer', '卖价', '卖出价', '报价'],
            'buy': ['buy', '买入', '买', '收'],
            'sell': ['sell', '卖出', '卖', '出']
        }
        
        # 添加常见的组合表达
        self.common_patterns = [
            "{SIDE} {PRODUCT}",
            "{SIDE} {PRODUCT} {YIELD}",
            "{SIDE} {PRODUCT} {QUANTITY}",
            "{PRODUCT} {SIDE} {YIELD}",
            "{PRODUCT} {YIELD} {SIDE}",
            "{SIDE} {PRODUCT} {YIELD} {QUANTITY}",
            "{PRODUCT} {QUANTITY} {SIDE} {YIELD}",
            "{PRODUCT} {YIELD} {QUANTITY} {SIDE}",
            "{SIDE} {PRODUCT} {QUANTITY} {DATE} {SPEED}"
        ]
    
    def generate_realistic_sample(self) -> Tuple[str, str]:
        """生成更真实的样本"""
        fields = self.generate_field_combination()
        
        # 使用常见模式生成
        pattern = random.choice(self.common_patterns)
        
        # 随机省略可选字段
        for field in ['YIELD', 'QUANTITY', 'DATE', 'SPEED']:
            if random.random() < 0.2 and field in fields:
                fields.pop(field)
        
        # 应用模式
        try:
            sentence = pattern.format(**fields)
        except KeyError:
            # 如果模式中包含的字段不存在，使用随机顺序
            field_items = list(fields.items())
            random.shuffle(field_items)
            words = []
            for field_name, field_value in field_items:
                words.extend(field_value.split())
            sentence = ' '.join(words)
        
        # 生成标注
        return self.sentence_to_tags(sentence)
    
    def sentence_to_tags(self, sentence: str) -> Tuple[str, str]:
        """将句子转换为标注"""
        words = sentence.split()
        tags = []
        
        for word in words:
            tag = self.classify_word(word)
            if tag:
                tags.append(tag)
            else:
                tags.append('O')
        
        return sentence, ' '.join(tags)
    
    def classify_word(self, word: str) -> str:
        """分类单个词"""
        # 方向字段
        if word in self.side_values:
            return f'S-SIDE'
        
        # 产品代码
        if word.replace('.', '').replace('IB', '').isdigit() and len(word) >= 6:
            return f'S-PRODUCT'
        
        # 收益率（包含小数点的数字）
        if '.' in word and word.replace('.', '').isdigit():
            return f'S-YIELD'
        
        # 数量（包含单位或纯数字）
        if any(unit in word for unit in self.quantity_units):
            return f'S-QUANTITY'
        if word.isdigit() and len(word) >= 3:
            return f'S-QUANTITY'
        
        # 日期
        if word in self.date_values:
            return f'S-DATE'
        
        # 速度
        if '+' in word and len(word) == 2 and word[1].isdigit():
            return f'S-SPEED'
        
        return None


def main():
    """主函数"""
    print("🚀 开始生成债券询价数据集...")
    
    # 创建基础生成器
    generator = BondQuoteDataGenerator()
    
    # 显示字段分布
    print("\n📊 字段分布统计（采样1000条）：")
    distribution = generator.get_field_distribution()
    for field, percentage in distribution.items():
        print(f"  {field}: {percentage:.1f}%")
    
    # 生成数据集
    print("\n📝 生成数据集...")
    generator.generate_dataset(num_samples=50000, output_dir='./generated_data')
    
    # 创建高级生成器
    print("\n🔧 生成更真实的样本...")
    advanced_generator = AdvancedDataGenerator()
    
    # 生成一些示例
    print("\n✨ 生成示例：")
    for i in range(10):
        sentence, tags = advanced_generator.generate_realistic_sample()
        print(f"\n示例 {i+1}:")
        print(f"句子: {sentence}")
        print(f"标注: {tags}")
    
    print("\n✅ 数据生成完成！")


if __name__ == '__main__':
    main()
