from train import Trainer  
from datasets import load_dataset


ds = load_dataset("fourteenBDr/toutiao")

full_data = ds['train']  # 假设 'train' 是数据集的名称
full_data = full_data.shuffle(seed=42)
split_data = full_data.train_test_split(test_size=0.2, seed=42)
train_data = split_data['train']
test_data = split_data['test']

trainer = Trainer(model_name="bert-base-chinese", 
                  train_data=train_data, 
                  test_data=test_data, 
                  max_len=32, 
                  batch_size=256, 
                  epochs=10,
                  save_dir='ckp'
                  )

trainer.train()



