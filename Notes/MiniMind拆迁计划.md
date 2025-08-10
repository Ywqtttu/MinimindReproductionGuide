#### **头比较痛的地方在于，严格地来说，我们现在只能算是，理解了建模的原理、大致流程、并且也许能够设计出自己的建模流程。**

#### **但是困难在于，当中的细节、尤其是训练细节，比如数据工程和调参，几乎是一片空白。**

这样，我们首先先分解它的项目结构，把训练代码里面，跟操纵模型无关的部分，我们先剥离出来，暂时忽略，然后先开始建模。

# 不对

现在我们要开始变换思路了。

- #### 阅读他的命令

- #### 然后查看命令调用的脚本文件

- #### 拆解脚本文件，把它分为一个个模块

- #### 自顶向下地以IPO模式分析模块们的依赖关系、作用、实现逻辑

- #### 然后再自下而上地逐步实现这些模块

# MiniMind拆迁计划

## 1. train_pretrain.py

### 1.1 main()

#### 1.1.1 处理流程

- parser定义各种超参数

- args存储parser中的超参数

- 实例化MiniMindConfig为lm_config
- 设置存储路径等
- 设置token字典的长度
- 初始化ctx
- 配置分布式训练
- 初始化wandb
- 初始化model和tokenizer
- 初始化PretrainDataset: train_ds
- 初始化分布式样例
- 初始化DataLoader: train_loader
- 初始化scaler
- 初始化optimizer
- 调用train_epoch, 在for循环中开始训练

#### 1.1.2 依赖函数

| Name                          | Description                            |
| ----------------------------- | -------------------------------------- |
| **argparse**                  | 初始化一个parser，设置并存储各种超参数 |
| **MiniMindConfig**            | Minimind模型设置                       |
| **ctx**                       | 一个能够提高计算效率的计算器           |
| **wandb**                     | 一个数据看板和模型优化器               |
| **init_model**                | 初始化model和tokenizer                 |
| **PretrainDataset**           | 实例化预训练数据集                     |
| **DistributedSampler**        | 分布式训练相关                         |
| **DataLoader**                | 生成数据迭代器                         |
| **torch.cuda.amp.GradScaler** | 一个控制模型参数提高优化效率的计算器   |
| **optim**                     | 优化器                                 |
| **train_epoch**               | 单轮训练方法                           |

### 1.2 argparse

- 使用argparse.ArgumentParser实例化一个parser，

然后使用

- add_argument(name_or_flags = , type = , default = )

可以逐个往解析器中添加参数。

- parser.parse_args() 可以返回所有参数，以类似字典的形式。

**无显式的依赖函数**

### 1.3 MiniMindConfig

继承了PretrainConfig类，用于设置模型的各种参数。

**无显式的依赖函数**

### 1.4 ctx

一个更高效的计算器，使用它计算会极大提高效率

**无显式的依赖函数**

### 1.5 wandb

算是一个数据看板+模型调优的小工具

**无显式的依赖函数**

### 1.6 init_model

初始化tokenizer和model，并且打印Log

| Name                | Description        |
| ------------------- | ------------------ |
| AutoTokenizer       | 实例化tokenizer    |
| MiniMindForCausalLM | 实例化MiniMind模型 |

### 1.7 AutoTokenizer

读取json文件，实例化一个tokenizer

**无显式依赖函数**

### 1.8 MiniMindForCausalLM

**实例化一个主模型，具体结构再议**

### 1.9 PretrainDataset

继承了torch的Dataset类，读取指定路径的文件，实例化数据集。

**无显式的依赖**

### 1.10 DistributedSampler

略。

### 1.11 DataLoader

接收一个Dataset的子类实例和其他相关参数，生成数据迭代器（其实也可以用手动for循环代替）

**无显式的依赖**

### 1.12 torch.cuda.amp.GradScaler

动态管理梯度缩放的一个扫描器，在参数优化的时候把optimizer放在这里边跑，好处多多。

**无显式的依赖**

### 1.13 optim

优化器，老生常谈

**无显式的依赖**

### 1.14 train_epoch

接收一个轮次，一个看板实例wandb

- 实例化交叉熵计算器
- 记录开始计算的时间
- 载入数据迭代器，for每一个step
  - 将数据放到GPU上
  - 更新学习率
  - for optimizer里的每一个param
    - 更新学习率
  - 使用ctx计算
    - 模型当前的残差
    - 模型当前的loss
  - 使用scaler后向传播优化参数
  - 假如经过了特定步数的训练，要进行梯度更新控制
    - 取消缩放
    - 裁剪梯度
    - 参数更新
    - 更新缩放器
    - 清空梯度
  - 假如经过了特定步数的训练，记录训练日志
    - 修正损失值
    - 获取学习率
    - 估算计算时间
    - 同步日志到wandb
  - 对于专家模型有另外安排，此处略。

| Name                 | Description  |
| -------------------- | ------------ |
| **CrossEntropyLoss** | 交叉熵计算器 |
| **get_lr**           | 更新学习率   |
| **scaler**           | 梯度缩放器   |
| **optimizer**        | 优化器       |
| **Logger**           | 记录日志     |

现在需要回头反思一下我们的方法论了

为什么，我们拆了半天还拆不出个所以然来？

对很多东西还都只有比较初级的了解。

现在要逐行复现PreTrain，大概2天内完成吧

我们还有三周左右。

有5个trainer、2个model

每一份代码花两天时间手敲复现+琢磨。

现在基本上情况是，每一部分都知道是干什么的了（对着代码）

但是无法独立地组装起来，并且独立编写。

**其实整个过程无非也就是以下几个操作：**

1. 初始化model
2. 初始化tokenizer
3. 初始化data_set
4. 初始化optimizer
5. 对data_set切片（或者生成迭代器）
6. 丢到一个循环里重复训练
   1. 算一下loss
   2. 更新下学习率
   3. 反向传播让optimizer更新一下参数

7. 训练完成

而对于我们这个项目而言，只是多了以下几个锦上添花的东西：

1. 人为设定了各种各样的超参数以约束训练规模或者控制训练质量
2. 引入了ctx/wandb/scaler/accumulation_step/grad_clip等优化训练效率或者辅助监督训练的实例
3. 在训练过程中时不时地记录一下训练成果

也没有特别的花里胡哨，但是实践起来仍然不容易，困难主要在于，对于torch等库的函数的熟练度、对训练程序的设计（比如上述提到的ctx等实例和各种超参数），还有训练前数据的准备。

**根据dpsk的反馈，所谓的“锦上添花”并非可有可无，这是训练工程强度之所在，是重要且困难的一环，同样需要重点掌握，由此可见，不仅仅设计模型结构重要，模型训练的设计也同样重要且困难，除此之外还有数据清洗工程，也同样重要且困难，这三大块是我们需要重点掌握的。**

## Model_MiniMind

### 1. MiniMindConfig





### 2. RMSNorm



### 3. Attention



### 4. FeedForward



### 5. MoEGate





### 6. MoEFeedForward





### 7. MiniMindBlock





### 8. MiniMindModel

持有一个config, 取出里面的参数

持有一个embed模型

持有一个dropout

持有一个layers

内含：num_hidden_layers个minimindBlock

持有一个norm

持有两个预计算好的freqs_cos, freqs_sin

持有2个register_buffer(两个缓存器，不会在反向传播中被更新)

**前向传播forward**





### 9. MiniMindForCausalLM

config_class = MiniMindConfig

self.config = MiniMindConfig

self.model = MiniMindModel

lm_head = Linear()

forward()

其实模型的设计还是在minimindmodel完成的，而这个ForCausalLM主要还是写出来供训练调用。







### 均值归一化



### RMSNorm

一种层归一化技术，通过均方根缩放激活值。

- 缓解内部协变量偏移

- 正则化
- 缓解梯度爆炸或者消失

### RoPE

对词向量的特定部分进行“旋转”来融入绝对位置信息，在计算注意力时体现出位置的特征。注意力机制可以自然地捕捉到词与词之间的相对位置关系，这对于理解序列的结构至关重要。



