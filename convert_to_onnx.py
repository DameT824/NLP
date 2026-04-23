import torch
from train_bert_ner import BertCRFNER
import torch.onnx

# 1. 加载训练好的模型
model = BertCRFNER(num_labels=30, bert_model_name='./models/bert-base-chinese')
model.load_state_dict(torch.load('bert_crf_bond_ner.pth', map_location='cpu'))
model.eval()

# 2. 创建示例输入（用于导出）
dummy_input_ids = torch.randint(0, 21128, (1, 64))  # batch_size=1, seq_len=64
dummy_attention_mask = torch.ones(1, 64, dtype=torch.long)

# 3. 导出BERT部分为ONNX（不包含CRF）
class BERTFeatureExtractor(torch.nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model.bert
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

extractor = BERTFeatureExtractor(model)

torch.onnx.export(
    extractor,
    (dummy_input_ids, dummy_attention_mask),
    "bert_bond_ner.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'seq_len'},
        'attention_mask': {0: 'batch_size', 1: 'seq_len'},
        'logits': {0: 'batch_size', 1: 'seq_len'}
    },
    opset_version=14
)

print("✅ BERT模型已导出为ONNX格式")

# 4. 导出CRF参数（单独保存为JSON）
crf_params = {
    'num_labels': model.crf.num_labels,
    'transitions': model.crf.transitions.detach().cpu().numpy().tolist(),
    'start_transitions': model.crf.start_transitions.detach().cpu().numpy().tolist(),
    'end_transitions': model.crf.end_transitions.detach().cpu().numpy().tolist()
}

import json
with open('crf_params.json', 'w') as f:
    json.dump(crf_params, f)

print("✅ CRF参数已保存")
