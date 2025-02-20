import torch
from model import BertClassify,inference
from transformers import AutoTokenizer
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d  - %(message)s'
)
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

model_name = 'bert-base-chinese'
hidden_size = 768
n_class = 15  
max_length = 128

model = BertClassify(model_name, hidden_size, n_class)

model_path = 'ckp/model_weights_epoch 5.pt'  

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))

tokenizer = AutoTokenizer.from_pretrained(model_name)

input_text = "突发！上海媒体曝出争议猛料：中超冠军创造耻辱性纪录，球迷痛骂"   

predicted_class = inference(model, tokenizer, input_text, max_length)

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

logging.info(f"目标文字: {input_text}")

logging.info(f"预测的中文类别: {chinese_category}")


