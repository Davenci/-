import torch
import torch.nn as nn
import torch.utils.data as Data
import deepspeed
from load_datasets import MyDataset  
from model import BertClassify  
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d  - %(message)s'
)
import os
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, classification_report

class Trainer:
    def __init__(self, model_name, train_data, test_data, max_len=64, batch_size=4, epochs=100, device=None, save_dir=None):
        self.model_name = model_name
        self.train_data = train_data
        self.test_data = test_data
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_dir = save_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize dataset and data loader
        self.train_data = Data.DataLoader(dataset=MyDataset(self.train_data, 
                                                            self.model_name,
                                                            self.max_len,),
                                          batch_size=self.batch_size, 
                                          shuffle=True, 
                                          num_workers=4)
        
        self.test_data = Data.DataLoader(dataset=MyDataset(self.test_data, 
                                                    self.model_name,
                                                    self.max_len,),
                                batch_size=self.batch_size, 
                                shuffle=False, 
                                num_workers=4)
        # Initialize model
        self.model = BertClassify(
                                    self.model_name, 
                                    hidden_size=768, 
                                    n_class=15         # 15个类别
                                  ).to(self.device)



        # Initialize optimizer and DeepSpeed
        self.model_engine, self.optimizer, _, _ = self.configure_deepspeed(self.model)



    def configure_deepspeed(self, model):
        # DeepSpeed配置文件
        deepspeed_config = {
            "train_batch_size": self.batch_size * 2,
            "gradient_accumulation_steps": 1,
            "fp16": {
                "enabled": True,
                "loss_scale": 128  # 启用混合精度训练
            },
            "zero_optimization": {
                "stage": 2,  # 启用ZeRO Stage 2
            },

            "gradient_clipping": 1.0 , # 设置梯度裁剪阈值

            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-5,
                    "weight_decay": 1e-2
                }
            },
            "local_rank": -1  # 自动分配 GPU
        }
        return deepspeed.initialize(model=model,
                                    model_parameters=model.parameters(),
                                    config=deepspeed_config, 
                                    )

    def train(self):
        sum_loss = 0
        total_step = len(self.train_data)
        for epoch in range(self.epochs):
            self.model.train()
            for i, batch in enumerate(self.train_data):

                input_ids = batch['input_ids'].to(self.device)

                attention_mask = batch['attention_mask'].to(self.device)

                labels = batch['label'].to(self.device)
                
                token_type_ids=batch['token_type_ids'].to(self.device)

                pred  = self.model(input_ids, token_type_ids, attention_mask)

                loss_fn = nn.CrossEntropyLoss() 
                loss = loss_fn(pred, labels)  
                sum_loss += loss.item() 

                self.model_engine.backward(loss)
                self.model_engine.step()

                if (i + 1) % 10 == 0:
                    logging.info(f'[{epoch + 1}/{self.epochs}] step:{i + 1}/{total_step} loss:{loss.item():.4f}')
            



            sum_loss = 0
        
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f'model_weights_epoch {epoch}.pt')  
            if dist.get_rank() == 0:  # 只让主进程保存检查点
                torch.save(self.model.state_dict(), save_path)
                print(f"权重保存在 {save_path}")

            accuracy, report = self.evaluate(self.test_data)
            logging.info(f'第{epoch}轮评估精确率: {accuracy}')
            logging.info(f'第{epoch}轮评估分类报告:\n{report}')

    def evaluate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in dataloader:
                if batch is None:
                    continue
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(input_ids, token_type_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds)

        return accuracy, report