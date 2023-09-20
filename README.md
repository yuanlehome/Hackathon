### 一、参考学习文档

#### IR&Pass Infra 相关

- IR Infra 开发 https://github.com/PaddlePaddle/Paddle/issues/55205

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



