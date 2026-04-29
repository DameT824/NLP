from train_bert_ner import BertCRFNER
import torch.onnx
import json
import os
from train_bert_ner import MODEL_PATH, OUTPUT_MODEL_PATH, WEIGHTS_PATH


# 标签映射
TAG2IDX = {'O': 0, '<PAD>': 1}
for field in ['SIDE', 'PRODUCT', 'YIELD', 'QUANTITY', 'DATE', 'SPEED']:
    for prefix in ['B', 'I', 'E', 'S']:
        TAG2IDX[f'{prefix}-{field}'] = len(TAG2IDX)

# 加载模型
model = BertCRFNER(len(TAG2IDX), bert_model_name=MODEL_PATH)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
model.eval()

# 准备示例输入
dummy_input = {
    'input_ids': torch.randint(0, 21128, (1, 64)),
    'attention_mask': torch.ones(1, 64, dtype=torch.long)
}

class BERTFeatureExtractor(torch.nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model.bert

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        return outputs.logits

extractor = BERTFeatureExtractor(model)
extractor.eval()

# 导出BERT模型为ONNX
print("正在导出BERT模型为ONNX...")
torch.onnx.export(
    extractor,
    (dummy_input['input_ids'], dummy_input['attention_mask']),
    f'{OUTPUT_MODEL_PATH}/bert_ner.onnx',
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size', 1: 'sequence_length'}
    },
    opset_version=14,
    verbose=False  #减少详细输出
)
print("BERT模型导出完成!")

onnx_size = os.path.getsize(f'{OUTPUT_MODEL_PATH}/bert_ner.onnx')
expected_min = 300 * 1024 * 1024
if onnx_size < expected_min:
    print(f"导出的ONNX模型大小为{onnx_size / 1024 / 1024:.2f}MB，小于预期最小值{expected_min / 1024 / 1024:.2f}MB，请检查导出过程！")
else:
    print(f"导出的ONNX模型大小为{onnx_size / 1024 / 1024:.2f}MB，已满足预期最小值{expected_min / 1024 / 1024:.2f}MB，导出成功！")

# 保存CRF参数
print("正在导出CRF参数...")
crf_params = {
    'num_labels': model.crf.num_labels,
    'trans_matrix': model.crf.trans_matrix.detach().numpy().tolist(),
    'start_trans': model.crf.start_trans.detach().numpy().tolist(),
    'end_trans': model.crf.end_trans.detach().numpy().tolist()
}

with open(f'{OUTPUT_MODEL_PATH}/crf_params.json', 'w', encoding='utf-8') as f:
    json.dump(crf_params, f, ensure_ascii=False, indent=2)
print("CRF参数保存完成!")

# 保存标签映射
with open(f'{OUTPUT_MODEL_PATH}/tag_mapping.json', 'w', encoding='utf-8') as f:
    json.dump({
        'tag2idx': TAG2IDX,
        'idx2tag': {v: k for k, v in TAG2IDX.items()}
    }, f, ensure_ascii=False, indent=2)
print("标签映射保存完成!")
