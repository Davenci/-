
# 文本分类模型项目说明
## 一、模型原理
本项目采用 Bert 模型构建文本分类模型。Bert（Bidirectional Encoder Representations from Transformers）是一种基于 Transformer 架构的预训练语言模型，具有强大的语言理解能力。
在本分类任务中，我们利用 Bert 模型最后一层输出的第一个 token 位置，即 CLS（Classification）位置的向量作为整个句子的表示。CLS 标记是在输入序列开始时添加的特殊标记，经过 Bert 模型处理后，其对应的输出向量会融合整个句子的语义信息。之后，将该向量输入到一个全连接层（Linear Layer）进行分类，全连接层会根据输入的特征向量输出每个类别的预测得分，通过后续的 softmax 函数将得分转换为概率，从而确定输入文本所属的类别。



## 二、训练步骤
### 2.1 环境安装
```
cd textclass
pip install -r requirements.txt
```
安装deepspeed的时候，建议单独用conda安装：
```
conda install mpi4py
```

### 2.2 数据集准备
本项目使用 HuggingFace 的开源数据集，该数据集涵盖了 15 种不同的中文文本类型，为模型提供了丰富多样的训练数据。你可以按照以下代码加载数据集：
```
from datasets import load_dataset
ds = load_dataset("fourteenBDr/toutiao")
```
### 2.3 训练参数配置
你可以在 
```
runner.py 
```
文件中设置训练所需的数据量和训练轮次。通过调整这些参数，你可以根据自己的需求和计算资源对模型进行训练优化。
```
# 设置所需的数据量
train_data_size = 10000
# 设置训练轮次
num_epochs = 5
```
### 2.4 分布式训练配置
为了加速模型训练过程，本项目采用了 DeepSpeed 框架进行分布式训练。DeepSpeed 是一个用于大规模分布式训练的优化库，它提供了多种训练策略和优化算法，可以有效提高训练效率和模型性能。
你可以在 
```
train.py 
```
文件中配置 deepspeed_config 来选择合适的训练策略。以下是一个简单的配置示例：
```
# train.py
import deepspeed

# 配置DeepSpeed训练参数
deepspeed_config = {
    "train_batch_size": self.batch_size * 2,
    "gradient_accumulation_steps": 1,
    "fp16": {
        "enabled": True,
        "loss_scale": 128  # 启用混合精度训练
    },
    "zero_optimization": {
        "stage": 2,  
    },

    "gradient_clipping": 1.0 , 

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-5,
            "weight_decay": 1e-2
        }
    },
    "local_rank": -1  # 自动分配 GPU
}

# 初始化DeepSpeed引擎
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=deepspeed_config
)
```
### 2.5 开始训练
在完成上述配置后，你可以通过运行 `textclass/train.sh` 脚本来启动训练过程。该脚本会自动加载数据集、配置训练参数并启动分布式训练。示例 train.
sh 脚本内容如下：
```
# train.sh
runner.py
```
在终端中执行以下命令开始训练：
```
train.sh
```
**模型在训练的时候，每一类结束都会对当前的权重进行评估测试**

通过以上步骤，你就可以完成基于 Bert 模型的文本分类任务的训练。在训练过程中，你可以根据实际情况调整参数和配置，以获得更好的模型性能。


### 2.6 权重保存
权重默认以PT格式自动保存在以下路径
```
ckp
```


### 2.7 模型推理
**PT格式权重推理**
替换脚本中`model_path`权重路径即可
```
inference.py
```

**onnx格式推理**
*权重转化*
在以下脚本中在`model_path`替换需要转化的权重
```
pt2onnx.py
```

*推理*
运行即可
```
inference_onnx.py
```