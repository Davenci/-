import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d  - %(message)s'
)

onnx_model = ort.InferenceSession("ckp_onnx/model.onnx")

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

input_text = "外资机构密集调研A股公司 紧盯AI与机器人产业机会"
inputs = tokenizer(input_text, return_tensors='np', padding=True, truncation=True, max_length=128)

input_ids = inputs['input_ids']
token_type_ids = inputs['token_type_ids']
attention_mask = inputs['attention_mask']

input_feed = {
    'input_ids': input_ids,
    'token_type_ids': token_type_ids,
    'attention_mask': attention_mask
}

onnx_outputs = onnx_model.run(None, input_feed)

predicted_class = np.argmax(onnx_outputs[0], axis=1)
predicted_class = predicted_class.item()


category_map = {
    100: 0,  # '故事'
    101: 1,  # '文化'
    102: 2,  # '环境'
    103: 3,  # 'news_sports'
    104: 4,  # 'news_finance'
    106: 5,  # 'news_house' 
    107: 6,  # 'news_car'
    108: 7,  # 'news_edu'
    109: 8,  # 'news_tech'
    110: 9,  # 'news_military'
    112: 10, # 'news_travel'
    113: 11, # 'news_world'
    114: 12, # 'stock'
    115: 13, # 'news_agriculture'
    116: 14  # 'news_game'
}
reverse_category_map = {v: k for k, v in category_map.items()}
original_category_id = reverse_category_map[predicted_class]

chinese_category_map = {
    100: '新闻故事',
    101: '新闻文化',
    102: '新闻娱乐',
    103: '新闻体育',
    104: '新闻财经',
    106: '新闻房产',
    107: '新闻汽车',
    108: '新闻教育',
    109: '新闻科技',
    110: '新闻军事',
    112: '新闻旅游',
    113: '新闻国际',
    114: '股票',
    115: '新闻农业',
    116: '新闻游戏'
}

chinese_category = chinese_category_map[original_category_id]

logging.info(f"预测的中文类别: {chinese_category}")