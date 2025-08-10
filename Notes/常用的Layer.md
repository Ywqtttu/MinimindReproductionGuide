### 线性层，也称为全连接层或密集层，是神经网络中最基本的组件之一。

作用 (Purpose):

线性层的主要作用是对输入数据执行**线性变换**。它学习其输入的加权总和并添加一个偏置项。这使得网络能够学习输入和输出之间的复杂关系。在大型模型中，线性层通常用于：

- **投影 (Projection):** 将输入特征投影到不同的维度空间（例如，增加或减少特征数量）。
- **变换 (Transformation):** 学习数据的抽象表示。
- **输出层 (Output Layer):** 在分类或回归任务中产生最终输出（通常在 Softmax 或 Sigmoid 激活函数之前）。

**常用场景 (Common Use Cases):**

- **Transformer 内部的前馈网络 (FFN within Transformers):** 用作逐位置前馈网络的一部分。
- **投影头 (Projection Heads):** 在像 BERT 这样的模型中，线性层用于将隐藏状态投影到词汇空间以进行掩码语言建模，或投影到较小空间以进行下一句预测。
- **输出层 (Output Layers):** 用于分类（例如，预测类别标签）或回归（预测连续值）。
- **嵌入变换 (Embedding Transformations):** 变换词嵌入或其他类型的嵌入。

简要的数学原理 (Brief Mathematical Principle):

线性层计算：

y=xWT+b

其中：

- x 是输入向量（或一批输入向量）。
- W 是权重矩阵。该矩阵中的值在训练期间学习得到。
- b 是偏置向量。该向量中的值也在训练期间学习得到。
- WT 是权重矩阵的转置。
- y 是输出向量。

维度通常如下：

- 如果 x 的形状是 (N,in_features)，其中 N 是批量大小。
- W 的形状是 (out_features,in_features)。
- b 的形状是 (out_features)。
- 那么 y 的形状将是 (N,out_features)。

**图像 (Conceptual Image):**

想象一组输入节点，每个节点都连接到每个输出节点。每个连接都有一个权重。

**核心要点：** 线性层对于学习变换至关重要，并且是将数据从一种表示映射到另一种表示的基础构建模块。

------

## ReLU (修正线性单元) 层

ReLU 是深度学习中使用最广泛的激活函数之一，尤其是在卷积神经网络 (CNN) 中，并且在大型语言模型 (LLM) 的前馈网络中也很普遍。

作用 (Purpose):

像 ReLU 这样的激活函数的主要目的是向模型引入非线性。没有非线性，神经网络无论有多少层，其行为都将像一个单层线性模型，这将严重限制其学习复杂模式的能力。

具体来说，ReLU：

- **引入非线性 (Introduces Non-linearity):** 允许模型学习比简单线性模型更复杂的函数。
- **解决梯度消失问题 (Addresses Vanishing Gradient Problem):** 对于正输入，梯度为 1，这有助于梯度在反向传播过程中比 sigmoid 或 tanh 函数更好地流动，后者可能会饱和。
- **计算效率 (Computational Efficiency):** 计算非常简单（**只是一个 `max(0, x)` 操作**），使得训练更快。
- **稀疏性 (Sparsity):** 它可能导致稀疏表示，因为神经元对于负输入可以输出零。这意味着一些神经元是“非活动的”，这有时可能有利于表示学习和效率。

**常用场景 (Common Use Cases):**

- **前馈网络中的隐藏层 (Hidden Layers in Feed-Forward Networks):** 通常在 Transformer 模型的逐位置前馈模块中的线性层之后使用。
- **卷积神经网络 (CNNs):** 在卷积层之后广泛使用。
- **通用深度神经网络 (General Deep Neural Networks):** 作为隐藏层的默认激活函数，除非有特定原因使用其他函数。

简要的数学原理 (Brief Mathematical Principle):

ReLU 函数定义为：

ReLU(x)=max(0,x)

这意味着：

- 如果 x>0，则 ReLU(x)=x
- 如果 x≤0，则 ReLU(x)=0

ReLU 的导数是：

- 如果 x>0，则为 1
- 如果 x<0，则为 0
- 在 x=0 处未定义（实际上，通常设置为 0 或 1）。

**图像 (Conceptual Image):**

该函数对于所有负输入均为零，对于正输入则线性增加。

**ReLU 的变体：**

- **Leaky ReLU:** 当单元未激活时，允许一个小的、非零的梯度 (LReLU(x)=max(0.01x,x))。有助于解决“ReLU 死亡”问题，即神经元可能卡在始终输出零的状态。

- **Parametric ReLU (PReLU):** 泄露在训练期间学习得到的系数。

- **Exponential Linear Unit (ELU):** 旨在使平均激活值更接近于零，这可以加速学习。

- ==**GELU (Gaussian Error Linear Unit):**== 通常用于 Transformer 模型（如 BERT、GPT）。

  它根据输入值对其进行加权，而不是像 ReLU 中那样根据其符号来门控输入。$ \text{GELU}(x) = x \Phi(x) $，其中 Φ(x) 是标准正态分布的累积分布函数。

虽然 ReLU 简单有效，但在最先进的大型语言模型中，由于经验上的性能改进，通常首选 GELU 等变体。

------

## Softmax 函数层

**Softmax 函数**是一个关键的激活函数，通常用于神经网络输出层的**多类别分类问题**。它将原始分数（logits）向量转换为概率分布。

### 作用 (Purpose):

- **概率分布 (Probability Distribution):** 将 K 个实数的向量转换为 K 个可能结果的概率分布。每个输出值都在 0 和 1 之间，并且所有输出值之和为 1。
- **多类别分类 (Multi-Class Classification):** 使模型能够为每个类别分配一个概率，指示输入属于该类别的可能性。通常选择概率最高的类别作为预测。
- **可解释的输出 (Interpretable Outputs):** 输出可以解释为每个类别的置信度分数。

### 常用场景 (Common Use Cases):

- **分类器的输出层 (Output Layer of Classifiers):** 最常见的用例。例如，在图像分类（预测图像是猫、狗还是鸟）中，或在自然语言处理（从词汇表中预测下一个词）中。
- **注意力机制 (Attention Mechanisms):** Softmax 用于注意力机制（例如，在 Transformer 中）以计算注意力权重。这些权重决定了对输入序列不同部分的关注程度。

### 简要的数学原理 (Brief Mathematical Principle):

给定一个由 K 个实数（logits）组成的输入向量 z=(z1,z2,...,zK)，Softmax 函数计算第 i 个类别的概率 pi 如下：

$$p_{i}=Softmax(z_{i})=\frac{e^{z_{i}}}{\sum^{K}_{j=1}e^{z_{j}}}$$

对于 i=1,...,K

### 数值稳定的 Softmax (Numerically Stable Softmax):

如果 zi 的值非常大（可能导致上溢）或非常小（可能导致下溢），直接计算 ezi 可能会导致数值不稳定。一个常见的技巧是在取指数之前从所有元素中减去 z 的最大值：

令 c=max(z1,z2,...,zK)。

$$p_{i}=Softmax(z_{i})=\frac{e^{z_{i-c}}}{\sum^{K}_{j=1}e^{z_{j-c}}}$$

这不会改变输出结果（因为分子和分母都乘以了 e−c），但能有效提高数值稳定性，避免计算溢出。PyTorch 等深度学习框架的实现内部通常处理了这个问题。

**图像 (Conceptual Image):**

将输入的 Logits 通过 Softmax 函数转换为输出概率，每个概率值介于0和1之间，且总和为1。

关于损失函数的重要说明：

在训练分类模型时，如果使用 torch.nn.CrossEntropyLoss，则不应将 Softmax 层作为模型的最后一层。CrossEntropyLoss 内部会计算 LogSoftmax 然后再计算 NLLLoss，因此它期望原始 logits 作为输入。在此之前应用 Softmax 是不正确的。但是，如果在推理过程中需要获得实际的类别概率，则会显式使用 Softmax。

------

## 嵌入层 (Embedding Layer)

嵌入层是神经网络中的一个关键组件，尤其用于处理分类数据，最著名的是自然语言处理 (NLP) 中的单词。它将离散的、高维的分类输入（如单词或用户 ID）映射到连续的、低维的密集向量表示，称为嵌入。

**作用 (Purpose):**

- **降维 (Dimensionality Reduction):** 将稀疏的、高维的独热编码向量（其中词汇量可能非常大）转换为密集的、低维的向量。这在内存和计算上都更加高效。
- **学习语义关系 (Learning Semantic Relationships):** 嵌入向量在训练过程中学习得到。理想情况下，语义相似的项（例如，具有相似含义或上下文的单词）将在向量空间中具有彼此靠近的嵌入向量。
- **特征学习 (Feature Learning):** 它充当分类数据的可学习特征提取器，允许模型发现有用的表示，而不是依赖于预定义的特征。

**常用场景 (Common Use Cases):**

- 自然语言处理 (NLP):
  - **词嵌入 (Word Embeddings):** 表示单词（例如，Word2Vec、GloVe，或在像 Transformer 这样的模型中从头开始学习）。词汇表中的每个单词都映射到一个向量。
  - **句子/文档嵌入 (Sentence/Document Embeddings):** 虽然不是直接用于整个句子的单个层，但嵌入层是第一步，之后是其他层（RNN、Transformer）来创建句子表示。
  - **位置嵌入 (Positional Embeddings):** 在 Transformer 中，为模型提供序列中标记位置的信息。
  - **标记类型嵌入 (Token Type Embeddings / Segment Embeddings):** 在像 BERT 这样的模型中，用于区分输入对中的不同句子。
- 推荐系统 (Recommender Systems):
  - **用户/物品嵌入 (User/Item Embeddings):** 将用户和物品表示为密集向量，以预测用户-物品交互（例如，评分、点击）。
- **分类特征处理 (Categorical Feature Handling):** 在任何处理分类输入的模型中（例如，产品类别、用户人口统计数据），其中独热编码会过于稀疏或维度过高。

简要的数学原理 (Brief Mathematical Principle):

从概念上讲，嵌入层就像一个查找表。

- 它维护一个形状为 

  (num_embeddings,embedding_dim)

   的嵌入矩阵。

  - `num_embeddings`: 词汇表的大小或唯一分类项的总数。
  - `embedding_dim`: 密集嵌入向量的期望维度。

- 当提供输入（表示类别/单词的整数索引）时，该层只是从此矩阵中“查找”并返回相应的行（嵌入向量）。

- 这些嵌入向量（嵌入矩阵的权重）被初始化（通常是随机的），然后在训练过程中通过反向传播进行更新，就像网络中的任何其他可学习参数一样。

如果输入是一个大小为 num_embeddings 的独热编码向量 x（在单词/类别的索引处为 1，其他地方为 0），并且嵌入矩阵为 E，则嵌入 v 为：

v=xE

然而，在实践中，为了提高效率，使用整数索引执行直接查找操作，而不是与独热向量进行矩阵乘法。

**图像 (Conceptual Image):**

输入一个整数索引（例如，词ID），嵌入层会从其内部的嵌入矩阵（查找表）中，根据这个索引找到对应的行，也就是该词的嵌入向量，并将其输出。

嵌入层对于使深度学习模型能够理解和处理离散的分类信息（尤其是文本）至关重要。

------

## 层归一化 (Layer Normalization / LayerNorm)

层归一化 (LayerNorm) 是一种归一化技术，它针对每个数据样本独立地对其**特征**维度上的输入进行归一化。这与批归一化 (Batch Normalization) 相反，后者是针对每个特征在批次维度上进行归一化。

**作用 (Purpose):**

- **稳定训练 (Stabilize Training):** 有助于稳定循环神经网络和 Transformer 中隐藏状态的动态，从而实现更平滑的梯度和更快的收敛。

- **减少协变量偏移 (Reduce Covariate Shift):** 与其他归一化技术类似，它有助于减少内部协变量偏移，即在训练过程中激活值分布发生变化的问题。

- 输入相关的归一化 (Input-Dependent Normalization):

   它根据每个样本自身特征的均值和方差来归一化该样本的激活值。这使其计算独立于批次中的其他样本，这对于以下情况特别有用：

  - **小批量大小 (Small Batch Sizes):** 其有效性不会因小批量大小而降低，这与批归一化不同。
  - **循环神经网络 (RNNs) 和序列数据 (Sequence Data):** 适用于长度可变的序列，其中批次统计数据可能没有意义或在时间步之间不一致。

- **有时能改善泛化能力 (Improved Generalization (Sometimes)):** 有时可以带来更好的泛化效果。

**常用场景 (Common Use Cases):**

- Transformer 模型 (Transformer Models):

   LayerNorm 是 Transformer 架构（例如 BERT、GPT）的基石。它通常应用于：

  - 多头注意力子层之前 (pre-LN)。
  - 前馈网络子层之前 (pre-LN)。
  - 或者在这些子层之后 (post-LN，如原始 Transformer 论文中那样)，尽管对于非常深的 Transformer，pre-LN 通常被认为更稳定。

- **循环神经网络 (RNNs):** 用于在 RNN 单元（例如 LSTM、GRU）内跨隐藏状态维度对激活值进行归一化。

- **强化学习 (Reinforcement Learning):** 有时用于策略网络或价值网络。

- **通用深度网络 (General Deep Networks):** 可以作为批归一化的替代方案，尤其是在批量较小或数据具有序列性质时。

简要的数学原理 (Brief Mathematical Principle):

对于给定的输入特征向量 x（对于批次中的单个样本，或该样本的某层激活值），LayerNorm 计算归一化输出 h 如下：

令 x=(x1,x2,...,xD) 为单个数据点的 D 维输入特征。

1. **计算特征间的均值 (\**μ\**)：** μ=D1i=1∑Dxi
2. **计算特征间的方差 (\**σ2\**)：** σ2=D1i=1∑D(xi−μ)2
3. **归一化：** xi−μ 其中 ϵ (epsilon) 是一个为数值稳定性而添加的小常数（以防止除以零）。
4. **缩放和移位 (可学习参数)：** 然后，归一化后的值通过一个可学习的参数 γ (gamma，也称为权重) 进行缩放，并通过一个可学习的参数 β (beta，也称为偏置) 进行移位。这些参数允许网络学习归一化是否确实有益，或者是否应将归一化后的值缩放/移位到不同的范围。 hi=γx^i+β γ 和 β 都是与 x 维度相同 (D) 的可学习参数向量。它们被初始化（例如，γ 初始化为 1，β 初始化为 0）并在训练期间更新。

**与批归一化的关键区别：**

- **LayerNorm:** ==对每个样本，跨其特征进行归一化。统计数据 (μ,σ2) 是为每个独立样本计算的。==
- **BatchNorm:** 对每个特征，跨批次中的样本进行归一化。统计数据 (μ,σ2) 是为每个特征跨整个批次计算的。

**图像 (Conceptual Image):**

考虑一批数据，其中每行是一个样本，每列是一个特征。LayerNorm 对每个样本的所有特征（即每一行）独立计算均值和方差，并进行归一化。

LayerNorm 对于训练像 Transformer 这样深而复杂的模型至关重要，它通过确保每一层内的激活值对于批次中的每个样本都保持良好分布来实现这一点。

------

## 批归一化 (Batch Normalization / BatchNorm)

批归一化 (BatchNorm) 是一种通过根据当前**小批量 (mini-batch)** 的统计数据（均值和方差）对层的输入进行重新中心化和重新缩放来归一化这些输入的技术。

**作用 (Purpose):**

- **减少内部协变量偏移 (Reduce Internal Covariate Shift):** 这是主要的动机。随着训练过程中前导层参数的变化，后续层输入的分布也会发生变化。BatchNorm 有助于稳定这些分布，使学习过程更平滑。
- **更快的收敛速度 (Faster Convergence):** 通过稳定分布和允许更高的学习率，BatchNorm 通常可以加快训练速度。
- **正则化效果 (Regularization Effect):** 由于使用小批量统计数据引入了噪声，BatchNorm 可能具有轻微的正则化效果，有时可以减少对 Dropout 等其他正则化技术的需求。
- **允许更高的学习率 (Allows Higher Learning Rates):** 使模型对参数的规模及其梯度的敏感性降低，从而能够使用更大的学习率。
- **减少对初始化的依赖 (Reduces Dependence on Initialization):** 使深度网络更容易初始化。

**常用场景 (Common Use Cases):**

- **卷积神经网络 (CNNs):** 在 CNN 中广泛使用，通常在卷积层之后和非线性激活函数之前应用（尽管有些人将其放在激活函数之后）。`nn.BatchNorm2d` 用于图像数据。
- **前馈神经网络 (FNNs):** 可以在线性层之后使用。`nn.BatchNorm1d` 用于全连接层。
- **通常不直接用于 RNN/LSTM 的循环激活 (Not Typically Used in RNNs/LSTMs directly on recurrent activations):** 对于序列，统计数据可能在时间步之间差异很大，批次统计数据可能不具有代表性。在这种情况下，通常首选层归一化 (Layer Normalization)。
- **在 Transformer 中不太常见 (Less Common in Transformers):** 虽然最初的 Transformer 没有使用 BatchNorm（它使用了 LayerNorm），但一些变体或特定应用可能会尝试使用它，但 LayerNorm 在大型语言模型中占主导地位。

简要的数学原理 (Brief Mathematical Principle):

对于一个包含 m 个样本的小批量，并考虑一个特定的特征（或 CNN 中的通道）：

1. **计算该特征的小批量均值 (\**μB\**)：** μB=m1i=1∑mxi 其中 xi 是小批量中第 i 个样本的该特征的值。
2. **计算该特征的小批量方差 (\**σB2\**)：** σB2=m1i=1∑m(xi−μB)2
3. **对每个样本的该特征进行归一化：** xi−μB 其中 ϵ 是一个为数值稳定性而添加的小常数。
4. **缩放和移位 (可学习参数)：** 然后，归一化后的值通过一个可学习的参数 γ (gamma，或权重) 进行缩放，并通过一个可学习的参数 β (beta，或偏置) 进行移位。这些参数特定于每个特征/通道，并允许网络学习归一化激活值的最佳缩放和移位。 yi=γx^i+β γ 和 β 在训练期间学习得到。γ 通常初始化为 1，β 初始化为 0。

**训练 vs. 推理 (During Training vs. Inference):**

- 训练期间：

   均值和方差是根据当前小批量计算的。此外，BatchNorm 层维护在训练期间观察到的激活值的均值和方差的

  运行估计值

  （移动平均值）。

  - running_mean=(1−momentum)×running_mean+momentum×μB
  - running_var=(1−momentum)×running_var+momentum×σB2 （注意：PyTorch 在其 running_var 更新中使用了略有不同的无偏方差估计公式，但概念是相同的）。

- **推理（评估）期间：** 在推理期间，使用小批量统计数据会使单个样本的输出依赖于其（可能是人为的）批次中的其他样本，这是不可取的。取而代之的是，使用训练期间学习到的均值和方差的**运行估计值**进行归一化。该层的行为类似于一个固定的线性变换。

**图像 (Conceptual Image):**

考虑一个小批量数据和一个特定特征（例如第 j 个特征）。BatchNorm 对该特征在所有样本中的值（即每一列）计算均值和方差，然后使用这些批次统计数据来归一化该特征的每个样本值。

**注意事项：**

- **依赖批量大小 (Batch Size Dependency):** 当批量非常小时，BatchNorm 的性能可能会下降，因为批次统计数据会变得嘈杂且不那么能代表整体数据分布。
- **并非适用于所有架构 (Not Ideal for All Architectures):** 如前所述，对于 RNN 和 Transformer，通常首选层归一化。
- 确保适当调用 `model.train()` 和 `model.eval()` 以在使用批次统计数据（并更新运行统计数据）和使用学习到的运行统计数据之间切换。

BatchNorm 是一项非常有影响力的技术，显著改善了深度 CNN 的训练。

------

## 随机失活层 (Dropout Layer)

Dropout 是一种正则化技术，用于防止神经网络中的过拟合。它通过在训练期间随机“丢弃”（即设置为零）一部分神经元的输出来工作。

**作用 (Purpose):**

- **减少过拟合 (Reduce Overfitting):** 这是主要目标。通过随机停用神经元，Dropout 可以防止复杂的协同适应，即神经元变得过度依赖于特定的其他神经元。这鼓励网络学习更鲁棒的特征，这些特征在与其他神经元不同随机子集结合使用时仍然有用。
- **近似集成效应 (Ensemble Effect (Approximation)):** 使用 Dropout 训练网络可以看作是近似训练大量“变瘦”的网络（即神经元子集的网络）的集成。在推理过程中，使用完整的网络（权重经过适当缩放）可以被认为是平均这些变瘦网络的预测。
- **改善泛化能力 (Improve Generalization):** 通过使网络对单个神经元的特定权重不那么敏感，Dropout 有助于其更好地泛化到未见过的数据。

**常用场景 (Common Use Cases):**

- **全连接层 (Dense Layers):** 非常普遍地在全连接层的激活函数之后应用 Dropout，尤其是在较深的网络中或观察到过拟合时。

- **卷积层 (Convolutional Layers):** 也可以应用于卷积层（`Dropout2d` 或 `Dropout3d` 会将整个通道置零），但它通常更常用于卷积块之后的全连接层。

- **循环神经网络 (RNNs):** 通常使用 Dropout 的特殊变体（如 `Variational Dropout` 或仅在非循环连接上应用 dropout）来防止跨时间步的信息丢失。应用于循环激活的标准 dropout 可能会阻碍学习长期依赖关系。PyTorch 的 `nn.LSTM` 和 `nn.GRU` 有一个 `dropout` 参数，可以在多层 RNN 的层与层之间正确应用它。

- Transformer 模型 (Transformer Models):

   Dropout 用于 Transformer 架构中的多个位置，例如：

  - 嵌入层之后。
  - 多头注意力输出之后。
  - 前馈网络输出之后。
  - 注意力权重上（注意力 dropout）。

**简要的数学原理 (Brief Mathematical Principle):**

训练期间：

对于应用了 Dropout 的层中每个神经元的输出 a：

- 以概率 p（dropout 率，例如 0.5），输出设置为 0（神经元被“丢弃”）。
- 以概率 1−p，输出保持不变。
- **缩放 (Scaling):** 为了确保下一层输入的期望总和在训练期间与推理期间（没有神经元被丢弃时）保持相同，那些*未被*丢弃的神经元的输出会按 1/(1−p) 的因子进行放大。这被称为**反向 dropout (inverted dropout)**，并且是最常见的实现方式（PyTorch 使用的方式）。

因此，对于一个激活值 x：

output={01−px以概率 p以概率 1−p

推理（评估/测试）期间：

Dropout 被关闭。所有神经元都被使用（没有输出被置零）。由于在训练期间应用了反向 dropout 缩放，因此在推理时不需要进一步缩放。网络按原样使用。

**图像 (Conceptual Image):**

训练期间 (每次迭代丢弃不同的神经元):

一个全连接的网络在训练的每次迭代中，会有一些连接（神经元）被随机“断开”。

推理期间 (所有神经元都激活，权重实际上已被缩放):

在推理时，网络恢复其所有连接，但由于训练时的反向缩放，其行为就像是多个“瘦”网络的平均。

**关键注意事项：**

- dropout 率 p 是一个需要调整的超参数。常见值范围从 0.1 到 0.5。
- 如果 p=0，则不应用 dropout。如果 p=1，则所有输出都置零（没有用）。
- 确保在训练周期之前调用 `model.train()`，在推理或验证之前调用 `model.eval()`，以正确启用/禁用 dropout 行为。这对于可复现性和正确的模型性能至关重要。

Dropout 是一种简单而强大的技术，已成为深度学习中对抗过拟合的标准工具。

------

## 注意力机制层 (Attention Layer) (特指自注意力 & 多头注意力)

注意力机制，特别是**自注意力 (Self-Attention)** 及其扩展**多头注意力 (Multi-Head Attention)**，是 Transformer 模型成功的基石，这些模型在自然语言处理领域取得了最先进的成果，并越来越多地应用于计算机视觉和语音识别等其他领域。

**作用 (Purpose):**

- 捕获上下文关系 (Capturing Contextual Relationships):

   允许模型在处理特定部分时权衡输入序列（或输入集）不同部分的重要性。

  - **自注意力 (Self-Attention):** 关注*同一*序列中的不同位置，以计算每个位置的表示。这有助于捕获长程依赖关系，并理解句子中单词/标记如何相互关联。例如，理解一个代词指的是哪个词。

- **处理可变长度输入 (Variable-Length Input Handling):** 可以有效地处理不同长度的序列。

- **可并行化 (Parallelizability):** 与逐个标记处理序列的 RNN 不同，自注意力可以同时计算序列中所有标记的注意力分数，从而能够在 GPU 上进行高效的并行化。

- **一定程度的可解释性 (Interpretability (to some extent)):** 注意力权重有时可以提供关于模型认为输入中哪些部分对给定输出很重要的见解，尽管这种解释应谨慎对待。

**常用场景 (Common Use Cases):**

- Transformer 模型 (核心组件):
  - **编码器层 (Encoder Layers):** 使用自注意力允许输入序列中的每个标记关注输入序列中的所有其他标记。
  - 解码器层 (Decoder Layers):
    - **掩码自注意力 (Masked Self-Attention):** 用于目标序列，其中一个标记只能关注先前的标记（及其自身），以防止在训练期间（自回归生成）从未来标记泄漏信息。
    - **交叉注意力 (Cross-Attention / Encoder-Decoder Attention):** 用于允许解码器中的标记（例如，目标语言句子）关注编码器输出的所有标记（例如，源语言句子）。这对于机器翻译等任务至关重要。
- **自然语言处理 (NLP):** 机器翻译、文本摘要、问答、情感分析、语言建模（例如 BERT、GPT）。
- **计算机视觉 (Computer Vision):** 视觉 Transformer (ViT) 使用自注意力来建模图像块之间的关系。
- **语音识别 (Speech Recognition):** 处理音频序列。

**简要的数学原理 (Brief Mathematical Principle):**

1. 缩放点积注意力 (Scaled Dot-Product Attention) (构建模块):

这是 Transformer 中最常用的注意力类型。

对于序列中的每个标记，我们根据从其输入嵌入（或前一层输出）派生的三个向量来计算其注意力输出：

\* 查询 (Query, Q): 当前标记的表示，询问“我在寻找什么？”

\* 键 (Key, K): 序列中所有标记（包括当前标记）的表示，充当值的标签或标识符。

\* 值 (Value, V): 序列中所有标记的表示，包含要聚合的实际信息。

过程如下：

a. 计算分数 (Calculate Scores): 计算当前标记的查询向量与序列中所有标记的键向量的点积。该分数表示查询与每个键的匹配程度。

scores=QKT

b. 缩放分数 (Scale Scores): 将分数除以键向量维度 dk 的平方根。这种缩放可以防止点积变得过大，从而可能将 softmax 函数推向梯度非常小的区域。

scaled_scores=dkQKT

c. (可选) 掩码 (Masking): 对于掩码自注意力（例如，在解码器中），在 softmax 之前，对应于未来位置的分数被设置为 −∞。这确保了一个标记不能关注后续标记。

d. 计算注意力权重 (Calculate Attention Weights): 对缩放后的分数沿键维度应用 Softmax 函数以获得注意力权重。这些权重总和为 1，并表示当前查询标记在所有值标记上的注意力分布。

AttentionWeights=softmax(dkQKT)

e. 计算输出 (Calculate Output): 将注意力权重乘以值向量并求和。这将产生值的加权和，其中权重由注意力分数确定。

AttentionOutput=AttentionWeights⋅V

通常，Q、K 和 V 是矩阵，其中每行对应序列中的一个标记。输入 X（例如，标记嵌入）通过学习到的线性变换（权重矩阵 WQ,WK,WV）投影到 Q、K、V：

Q=XWQ  K=XWK  V=XWV

2. 多头注意力 (Multi-Head Attention):

多头注意力不是用 dmodel 维的 Q、K、V 执行单个注意力函数，而是并行运行多个缩放点积注意力操作（“注意力头”）。

a. 输入的 Q、K、V 通过不同的、学习到的线性投影被线性投影 h 次（头的数量）到较小的维度（dq,dk,dv，其中 dk=dv=dmodel/h）。

b. 对每个头独立执行缩放点积注意力，产生 h 个输出矩阵。

c. 这 h 个输出矩阵被连接起来，然后再次通过另一个学习到的权重矩阵 WO 进行线性投影，以产生多头注意力层的最终输出。

**多头的好处：**

- 允许模型在不同位置联合关注来自不同表示子空间的信息。单个注意力头可能被迫对不同类型的信息进行平均。
- 每个头可以学习不同类型的关系（例如，一个头用于句法关系，另一个头用于语义相似性）。

**图像 (Conceptual Image):**

缩放点积注意力：

输入序列中的每个词（例如 "cat"）会生成一个查询向量 (Q_cat)。同时，序列中的所有词（"The", "cat", "sat"）都会生成键向量 (K_The, K_cat, K_sat) 和值向量 (V_The, V_cat, V_sat)。Q_cat 与所有的 K 进行点积、缩放、Softmax 得到注意力权重。这些权重随后用于对 V 进行加权求和，得到 "cat" 这个位置的最终注意力输出。

多头注意力：

输入首先通过多个并行的线性投影层，分别生成对应不同“头”的 Q, K, V。每个头独立执行缩放点积注意力计算。然后，所有头的输出被拼接起来，再通过一个最终的线性投影层得到最终输出。

注意力层，尤其是多头自注意力，每个交互的计算量比简单的线性层或卷积层要大，但它们捕获全局依赖关系的能力以及其可并行性使其成为大型模型的不可或缺的组成部分。

------

## Transformer 前馈网络层 (Transformer Feed-Forward Network / FFN Layer)

前馈网络 (FFN) 是 Transformer 模型每一层（编码器和解码器堆栈中均有）内部的一个子组件。在注意力机制之后，它独立地应用于序列中的每个位置（标记）。

**作用 (Purpose):**

- **非线性和转换 (Non-linearity and Transformation):** 为模型提供额外的非线性和处理能力。在注意力机制聚合了来自序列不同部分的信息之后，FFN 会进一步处理每个标记的这种聚合表示。
- **增加模型容量 (Increased Model Capacity):** 添加更多可学习的参数，增加 Transformer 层的整体容量以建模复杂函数。
- **映射到不同的表示空间 (Mapping to a Different Representational Space):** 它通常会扩展表示的维度，然后再将其投影回来。这种扩展-压缩模式被认为允许模型学习更丰富、更多样化的特征。

**常用场景 (Common Use Cases):**

- **Transformer 编码器层 (Transformer Encoder Layers):** 在自注意力子层（及其相关的残差连接和层归一化）之后应用。
- **Transformer 解码器层 (Transformer Decoder Layers):** 在掩码自注意力和编码器-解码器交叉注意力子层（及其相关的残差连接和层归一化）之后应用。
- 它是 Transformer 架构（如 "Attention Is All You Need" 中定义）的标准和组成部分，并在几乎所有后续的基于 Transformer 的模型（BERT、GPT 等）中使用。

简要的数学原理 (Brief Mathematical Principle):

FFN 通常是一个两层全连接神经网络，中间带有一个非线性激活函数。对于输入 x（这是前一个注意力子层的输出，经过层归一化和残差连接后，针对特定标记）：

1. **第一个线性变换 (扩展):** Linear1(x)=xW1+b1 其中 W1 是一个权重矩阵，通常将输入维度 dmodel 扩展到一个内部层维度 dff (例如，dff=4×dmodel)。b1 是偏置项。
2. **非线性激活:** 应用像 ReLU (修正线性单元) 或 GELU (高斯误差线性单元) 这样的激活函数，逐元素进行。GELU 在现代 Transformer 如 BERT 和 GPT 中更为常见。 Activation(Linear1(x))
3. **第二个线性变换 (投影回 \**dmodel\**):** FFN(x)=(Activation(xW1+b1))W2+b2 其中 W2 是一个权重矩阵，将 dff 维表示投影回 dmodel。b2 是偏置项。

这整个 FFN 操作是**逐位置 (position-wise)** 应用的，意味着相同的 FFN（具有相同的 W1,b1,W2,b2）独立地应用于序列中每个标记的表示。虽然参数在序列中的不同位置共享，但它们对于每个 Transformer 层是不同的。

**图像 (Conceptual Image):**

对于单个标记的表示 x（维度为 dmodel）：

输入 x（来自注意力子层的输出，针对一个标记），维度为 dmodel。

首先通过第一个线性层（权重 W1，偏置 b1），扩展到维度 dff（例如 4×dmodel）。

然后通过激活函数（ReLU/GELU）。

最后通过第二个线性层（权重 W2，偏置 b2），投影回维度 dmodel，得到该标记的输出。

这个模块对序列中的每个标记都独立应用，但使用相同的权重 W1,b1,W2,b2。

FFN 虽然结构简单（两个线性变换加一个激活函数），但在 Transformer 中扮演着至关重要的角色，它在注意力机制混合了上下文信息之后，为每个标记独立地提供了必要的非线性处理和表示能力。