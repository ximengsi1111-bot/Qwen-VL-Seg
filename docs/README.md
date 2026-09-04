# Qwen3-VL-Seg 从零复现项目文档

本文档是 Qwen3-VL-Seg 从零复现项目的入口。当前阶段采用小数据策略，先在 4 卡 A800 上验证从环境、数据、模型、训练到评测的完整链路，再根据结果决定是否扩展数据和卡数。

## 环境命名

- 本地电脑：`kk`
- 当前服务器：`祝融`（`172.29.15.117`，主机名 `choud-mgt-09`）
- 租赁服务器：`DL`（2 × RTX 5090D 32GB，Blackwell `sm_120`）

## 项目路径

- 本地代码目录（`kk`）：`D:\XYK\Qwen-VL-Seg`
- 祝融代码目录：`/mingli01/project/xiyongkai/qwen3vl-seg`
- 祝融资产目录：`/file_storage01/home/mingli/data/xyk`
- 祝融登录节点：`172.29.15.117`（`choud-mgt-09`）
- 当前使用 Slurm 分区：`gre`
- 调试阶段固定使用：4 卡

## 文档索引

- `GLOBAL_PLAN.md`：全局复现目标、范围、架构、里程碑和运行原则
- `01_ENVIRONMENT.md`：服务器环境、conda 环境、依赖、Slurm 提交方式
- `02_DATA.md`：统一数据 schema、公开数据、SA1B-ORS、ORS-Bench 小集
- `03_MODEL.md`：Qwen3-VL-4B 接入和 Box-Guided Mask Decoder 实现
- `04_TRAINING.md`：Stage1/Stage2 训练、分布式、checkpoint 和日志
- `05_EVALUATION.md`：RES/REC、ORS、通用多模态评测和指标计算

## 当前状态

- 已完成远端代码目录与资产目录创建
- 已完成旧 `segment` 环境删除，并创建同名 Python 3.10 环境
- 已确认 `gre` 分区、`module load conda/3 cuda/13.0`、torchrun/accelerate 提交方式
- 尚未开始正式环境依赖安装、数据构建、模型代码和训练

> 模块状态：`01-05` 均处于 `planned`（方案已定，尚未实现）。

## 命名与密钥约定

- Python 包入口统一为：`qwen3vl_seg.{model,data,train,eval}.<module>`。
- 模型主类：`qwen3vl_seg.model.model_wrapper.Qwen3VLSegForSegmentation`。
- 解码器：`qwen3vl_seg.model.mask_decoder.BoxGuidedMaskDecoder`。
- 数据转换入口：`qwen3vl_seg.data.build_public / build_sa1b_ors / build_ors_bench`。
- 训练入口：`qwen3vl_seg.train.training_loop`。
- 评测入口：`qwen3vl_seg.eval.runner`。
- `.env`、`configs/secrets/*.yaml` 均不提交并设置仅 owner 可读；密钥只能从环境变量或 secret 文件读取，不得写入日志。

## 小数据规模口径（最终版）

- 公开数据 train：约 6000 条样本（RefCOCO 系 3000、COCO 1500、LVIS 1500，允许 ±5%）。
- 公开数据 val / test：各约 600 条（RefCOCO 系 val/test 各 200×3）。
- SA1B-ORS 小集：CoRS 与 DeRS 各至少 300 条。
- ORS-Bench 小集：ID 400 条、OOD 300 条。
- 训练步数：Stage1 500 步、Stage2 200 步；smoke 为 Stage1 100 步、Stage2 50 步。

## 执行原则

1. 不复用现有 `Qwen3-VL-Seg` 项目代码，仅参考服务器使用方式。
2. 所有新代码进入远端 `qwen3vl-seg`；数据、权重、结果进入 `$DATA_ROOT`。
3. 默认不写 `/persist_data`，避免撑爆共享盘。
4. 调试阶段最多 4 卡；正式训练阶段才扩展至 20 卡。
5. 每个里程碑先跑 smoke test，通过后再进入下一阶段。
