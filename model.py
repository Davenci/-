import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d  - %(message)s'
)

class BertClassify(nn.Module):
    def __init__(self,
                 model_name, 
                 hidden_size, 
                 n_class
                 ):
        
        """
        model_name: 预训练模型 bert
        hidden_size：模型处理维度
        n_class：类别数量
        """
        super(BertClassify, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=True, return_dict=True)
        self.linear = nn.Linear(hidden_size, n_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        X: 字典类型，包含 input_ids, attention_mask, token_type_ids
        """

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = self.linear(self.dropout(outputs.pooler_output))
        return logits
    


def inference(model, tokenizer, input_text, max_length=128, device='cpu'):
    # 对输入文本进行预处理
    encoding = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)



    # 将模型设置为评估模式
    model.eval()
    with torch.no_grad():
        # 进行推理
        logits = model(input_ids,attention_mask, token_type_ids)
        # 获取预测的类别索引
        predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class



