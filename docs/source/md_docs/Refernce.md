# 理论资料
相关论文资料与理论知识集合

_参考连接_：

## 模型压缩相关资料


## 代码论文与相关资料

论文原文地址：
- siamban:
   - [论文原文](https://arxiv.org/abs/2003.06761)
   - [code](https://github.com/hqucv/siamban)
   
相似论文:
- siamcar: 
   - [论文原文](https://arxiv.org/abs/1911.07241)
   - [code](https://github.com/ohhhyeahhh/SiamCAR)


## 改进意见
1. 将[resnet50](https://github.com/IRLSCU/siamban/blob/20210414_model-amend/siamban/models/backbone/resnet_atrous.py)中的1，2层基础单元更改为ShuffleNetv2基础结构，2，3层使用了空洞卷积，也可以使用 ShuffleNetv2;
2. 
