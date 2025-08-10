### train_gpt(model, optimizer, data_loader, max_iters = 1145)

for(max_iters){

​	梯度清空;

​	加载数据;

​	model(inputs).logits;

​	转置logits;

​	计算loss(交叉熵);

​	后向传播; 优化模型参数;

​	输出当前loss

}

### model: gpt-2 + LoRA

这里的model是被peft掺入了LoRA的结果。

### model(inputs)-返回输出

那么其实主要的步骤还是在设计peft()这个类，然后把它传递给要微调的model。

# 接下来总结一下LoRA微调的流程（假设是向某垂类领域微调）

刘老师好，我又来了嘿嘿

上回跳过了唐老师的LLM部分去找了个机构的教程学，就是上面给您看的那个，结果他们实际的流程到一半就断了，只讲完了用Langchain部署大模型做各种小的代理应用之后就断了，甚至没开始讲模型微调和私有化部署，然后我又回去看唐老师的LLM部分了，是全看完了，可惜LoRA及之后的视频还没有录完，但是我发现Github的仓库里是有之后部分的代码及简单的文档的。

后来我又去找了个教程，这个教程的“微调”是在阿里云或者智谱等类似平台上进行了，他们简化了微调的过程，只需要交钱、上传数据就可以练了。

我有点问题想向您请教

就是我大概总结了一下唐老师的LoRA部分(ch11_llm)里微调的流程：

1. 准备数据集（包含有data和labels)
2. **定义peft方法，主要是设计好LoraCoinfig，把loraConfig和model结合。也就是往模型里增加一些LoRA参数**

3. 然后就是一般模型训练的流程（一个示例如下）
   1. 定义评价模型的类或者函数（假如需要的话）
   2. 定义优化器optimizer，传入model的参数
   3. 计算loss
   4. 传递loss给optimizer
   5. 后向传播调参

而阿里云或者智谱这样的平台只是把上述部分的代码封装了，然后做了个GUI的前端接收输入？

以及，想问问老师，唐老师仓库([regression2chatgpt/ch11_llm at zh · GenTang/regression2chatgpt](https://github.com/GenTang/regression2chatgpt/tree/zh/ch11_llm))里代码的结构，是否就是现在学术界或者企业界真的在用的微调模型的一种常用的范式，是可以让我直接拿来模仿、使用的？还是说这只是一个“特供”给新手的版本，和大家正在用的方法并不一样？







