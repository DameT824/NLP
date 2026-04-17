# evaluate_bert_ner.py
import torch
from train_bert_ner import BertCRFNER, BondQuoteDataset
from transformers import BertTokenizer
from sklearn.metrics import classification_report, f1_score


def detailed_evaluation():
    """详细评估"""
    # 加载模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    test_dataset = BondQuoteDataset(
        './generated_data/test.words.txt',
        './generated_data/test.tags.txt',
        tokenizer
    )
    
    num_labels = len(test_dataset.tag2idx)
    model = BertCRFNER(num_labels)
    model.load_state_dict(torch.load('bert_crf_bond_ner.pth'))
    model.eval()
    
    # 收集预测结果
    all_predictions = []
    all_labels = []
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            predictions = model(input_ids, attention_mask)
            
            for i in range(len(predictions)):
                mask = attention_mask[i].bool()
                preds = predictions[i][:mask.sum()]
                labs = labels[i][:mask.sum()]
                all_predictions.extend(preds.numpy())
                all_labels.extend(labs.numpy())
    
    # 生成分类报告
    tag_names = [test_dataset.idx2tag[i] for i in range(len(test_dataset.idx2tag))]
    report = classification_report(
        all_labels, all_predictions, 
        target_names=tag_names, 
        zero_division=0
    )
    
    print("📊 模型评估报告：")
    print(report)
    
    # 计算实体级F1
    entity_f1 = calculate_entity_f1(all_labels, all_predictions, test_dataset.idx2tag)
    print(f"\n实体级F1分数: {entity_f1:.4f}")


def calculate_entity_f1(true_labels, pred_labels, idx2tag):
    """计算实体级F1"""
    def extract_entities(labels):
        entities = []
        current_entity = None
        
        for idx, label_idx in enumerate(labels):
            tag = idx2tag[label_idx]
            if tag.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {'type': tag[2:], 'start': idx}
            elif tag.startswith('I-') and current_entity:
                continue
            elif tag.startswith('E-') and current_entity:
                current_entity['end'] = idx
                entities.append(current_entity)
                current_entity = None
            elif tag.startswith('S-'):
                if current_entity:
                    entities.append(current_entity)
                entities.append({'type': tag[2:], 'start': idx, 'end': idx})
                current_entity = None
            elif tag == 'O' and current_entity:
                current_entity['end'] = idx - 1
                entities.append(current_entity)
                current_entity = None
        
        if current_entity:
            current_entity['end'] = len(labels) - 1
            entities.append(current_entity)
        
        return entities
    
    true_entities = extract_entities(true_labels)
    pred_entities = extract_entities(pred_labels)
    
    # 计算精确率、召回率、F1
    true_set = set((e['type'], e['start'], e['end']) for e in true_entities)
    pred_set = set((e['type'], e['start'], e['end']) for e in pred_entities)
    
    if len(pred_set) == 0:
        return 0.0
    
    precision = len(true_set & pred_set) / len(pred_set)
    recall = len(true_set & pred_set) / len(true_set) if len(true_set) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


if __name__ == '__main__':
    detailed_evaluation()
