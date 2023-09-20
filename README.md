### 零、环境准备

[Ai-studio开发机器申请链接](https://github.com/PaddlePaddle/community/tree/master/pfcc/call-for-contributions#%E9%A3%9E%E6%A1%A8%E7%BA%BF%E4%B8%8A%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83ai-studio)

### 一、参考学习文档

#### IR&Pass Infra 相关

- IR Infra 设计 https://github.com/PaddlePaddle/Paddle/issues/55205

- Pass Infra 开发 https://github.com/PaddlePaddle/Paddle/pull/54738 以及 PR 描述关联链接

- DRR 开发 https://github.com/PaddlePaddle/Paddle/pull/55859

#### Phi 算子库及定义 Codegen 相关

- [飞桨高可复用算子库 PHI 设计文档](https://github.com/PaddlePaddle/docs/blob/develop/docs/design/phi/design_cn.md)
- [OpKernel 迁移指南](https://github.com/PaddlePaddle/docs/blob/develop/docs/design/phi/kernel_migrate_cn.md)
- [算子定义生成体系建设--静态图算子自动生成](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/paddle_autogen_code.md)
- [算子定义生成体系建设--静态图算子自动生成-第二期](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/paddle_autogen_code_2.md)

### 二、工作内容

#### Fuse kernel 迁移至 Phi 与新 IR Fuse op 定义自动生成

| Op                                                    | 后端 | PR链接                                            | 优先级 | 备注 |
| ----------------------------------------------------- | ---- | ------------------------------------------------- | ------ | ---- |
| multihead_matmul                                      | gpu  | https://github.com/PaddlePaddle/Paddle/pull/56846 | P0     | done |
| conv2d_fusion                                         | gpu  |                                                   | P0     |      |
| skip_layernorm                                        | gpu  |                                                   | P0     |      |
| fused_multi_transformer                               | gpu  |                                                   | P0     |      |
| multihead_matmul_roformer                             |      |                                                   | P0     |      |
| fused_embedding_eltwise_layernorm                     |      |                                                   | P0     |      |
| fuse_eleadd_transpose                                 |      |                                                   |        |      |
| layernorm_shift_partition                             |      |                                                   |        |      |
| fusion_transpose_flatten_concat                       |      |                                                   |        |      |
| fusion_repeated_fc_relu                               | cpu  |                                                   |        |      |
| layernorm_shift_partition                             |      |                                                   |        |      |
| 待梳理 paddle/fluid/operators/fused 目录下所有fuse op |      |                                                   |        |      |



#### Pass 推全
1. 


#### 常量折叠 Pass 与新 IR Infra 关键机制设计与实现
1. 根据常量折叠Pass的需求对新IR中Parameter的机制与管理方式重新进行设计（需要熟悉训练和推理场景下IR Program的加载、保存、执行等关键流程）
2. 设计机制支持Op在Pass阶段进行常量计算

#### 新IR Pass 推理单测基础设施搭建
在Python端打通新IR跑模型推理的流程（包括：模型加载，模型转换为新IR，Pass优化，模型推理执行），并封装为通用接口用于后续Pass推全时进行模型的批量验证。
