import os
import torch
import torch.onnx
from model import BertClassify
from transformers import AutoTokenizer
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d  - %(message)s'
)


# 模型配置
model_name = 'bert-base-chinese'
hidden_size = 768
n_class = 15  

# 初始化模型
model = BertClassify(model_name, hidden_size, n_class)

model_path = 'ckp/model_weights_epoch 5.pt'


model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
model.eval()


tokenizer = AutoTokenizer.from_pretrained(model_name)


input_text = "外资机构密集调研A股公司 紧盯AI与机器人产业机会"
inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=128)

# 提取输入张量
input_ids = inputs['input_ids']
token_type_ids = inputs['token_type_ids']
attention_mask = inputs['attention_mask']


save_dir = "ckp_onnx"
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir,"model.onnx")

onnx_path = save_path

torch.onnx.export(model, 
                  (input_ids, token_type_ids, attention_mask),  
                  onnx_path, 
                  export_params=True, 
                  opset_version=14,  
                  input_names=['input_ids', 'attention_mask', 'token_type_ids'],  
                  output_names=['output'],  
                  dynamic_axes={'input_ids': {0: 'batch_size'}, 'attention_mask': {0: 'batch_size'}, 'token_type_ids': {0: 'batch_size'}, 'output': {0: 'batch_size'}})  # 动态维度
