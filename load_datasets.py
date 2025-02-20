import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d  - %(message)s'
)


class MyDataset(Dataset):
    def __init__(self, dataset, model_name, max_len=64):
        self.model_name = model_name
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 使用datasets库加载数据
        self.data = dataset
        self.category_map = {
                            100: 0,  # 'news_story'
                            101: 1,  # 'news_culture'
                            102: 2,  # 'news_entertainment'
                            103: 3,  # 'news_sports'
                            104: 4,  # 'news_finance'
                            106: 5,  # 'news_house' (跳过 105)
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
        
        self.reverse_category_map = {v: k for k, v in self.category_map.items()}
        # logging.info(f"self.reverse_category_map {self.reverse_category_map}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 从datasets中获取样本
        sample = self.data[idx]
        # logging.info("-----------------------一组样本---------------------")
        # logging.info(f"样本 {sample}")
        parts = sample['text'].split('_!_')
        # logging.info(f"parts {parts}")

        # 提取新闻标题（第4个字段）和分类code（第2个字段）
        news_title = parts[3]  # 新闻标题
        category_code = int(parts[1])

        label = self.category_map.get(category_code)

        # logging.info(f"分类标签 {label}")
        # logging.info(f"分类内容 {news_title}")
        
        # 如果 category_code 不在 category_map 中，跳过该样本或处理
        if label == -1:
            logging.warning(f"未知的分类代码 {category_code} 跳过该样本")
            return None  # 或者处理成其他的标记

        
        # 使用BERT tokenizer处理新闻标题
        encoding = self.tokenizer(news_title,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len,
                                  return_tensors='pt')
        
        input_ids = encoding['input_ids'].squeeze(0) # [max_len]
        attention_mask = encoding['attention_mask'].squeeze(0)  # [max_len]
        token_type_ids = encoding['token_type_ids'].squeeze(0)


        return {
            # "text":news_title,
            # "category_code":category_code,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label,
            'token_type_ids': token_type_ids
        }


if __name__ == '__main__':
    # 加载数据集
    ds = load_dataset("fourteenBDr/toutiao")

    # 获取训练集数据
    train_data = ds['train']  # 假设 'train' 是数据集的名称

    # 创建自定义数据集
    dataset = MyDataset(dataset=train_data, model_name='bert-base-chinese', max_len=64)

    # 使用DataLoader进行批处理
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)

    # 查看第一个批次
    sample = next(iter(train_loader))
    logging.info(sample)
