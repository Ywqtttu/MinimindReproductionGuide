# MiniMind复现指南
基于jingyaogong大佬的MiniMind仓库（https://github.com/jingyaogong/minimind），探讨如何从0开始复现并训练一个轻量化LLM（其实就是MiniMind-Dense）。
# **注意**
由于笔者本地算力所限，所有源码均只在Colab上运行过，下载到本地未必能即点即用。
这其实只算得上对大佬模型的拙劣模仿和复现，实际上只做到了完成预训练那步，微调和MOE版模型因为笔者期末考的缘故中道崩殂了，没有完全做完，因此就没有上传了。
# 0810紧急补丁
* Codes_pure这个文件夹本来是留着存可命令行交互的纯代码的，但是笔者命令行相关库功夫没学到家（），一时不能完成，因此暂时置空。
* 优化了可读性、更换了训练集。
# 各文件夹说明
## BreakdownProcess
分解和复现MiniMind（MiniMind-Dense）的完整文件夹，包含对每个组件的解读和可运行的预训练脚本（即./BreakdownProcess/trainer中的Pretrain.ipynb）。
## BreakdownGPT
基于复旦唐老师的教程对GPT的简单复现，这里更多的是完整版的代码，而没有完整保留摸索阶段的痕迹。
## TorchDemo
笔者学习torch时留下的痕迹。
## Notes
笔者的学习笔记。
