# 环境搭建详细方案

> 状态：planned（尚未安装依赖、尚未首次提交 GPU 作业）

## 0. 环境命名

- 本地电脑：`kk`
- 当前服务器：`祝融`（`172.29.15.117` / `choud-mgt-09`）
- 租赁服务器：`DL`（2 × RTX 5090D 32GB，Blackwell `sm_120`）

## 1. 目标

在祝融服务器（`172.29.15.117`，登录节点 `choud-mgt-09`）上建立可复现的 Python 环境，并打通 4 卡 Slurm 提交链路。

## 2. 服务器现状

- 登录节点无 GPU，但安装 Slurm。
- Slurm 分区：`gre`。
- 调试阶段固定使用 4 卡：`--gres=gpu:4 --nodes=1`。
- conda：`/persist_data/apps/miniconda3/condabin/conda`。
- 已创建同名环境：`segment`，Python 3.10.21。
- 可用 module：`conda/3`、`cuda/13.0`。
- 磁盘约束：
  - `/persist_data`：大约 17GB 可用，不作为主存储。
  - `/mingli01`：约 3.3TB 可用。
  - `/file_storage01`：约 12TB 可用。
  - `/file_storage02`：约 61TB 可用，几乎为空。

## 3. 目录约定

```text
/mingli01/project/xiyongkai/qwen3vl-seg
├── configs/
├── qwen3vl_seg/
├── scripts/
├── tests/
├── docs/
├── requirements.in
└── requirements.lock

/file_storage01/home/mingli/data/xyk
├── datasets/
├── checkpoints/
├── runs/
└── logs/
```

环境变量模板统一提交为 `qwen3vl-seg/.env.example`，实际运行文件 `qwen3vl-seg/.env` 不提交；两者都不含密钥：

```bash
export DATA_ROOT=/file_storage01/home/mingli/data/xyk
export PROJECT_ROOT=/mingli01/project/xiyongkai/qwen3vl-seg
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## 4. Python 依赖

### 4.1 核心依赖

```text
torch==2.4.0+cu121
torchvision
transformers>=4.57,<5
peft
accelerate
deepspeed
safetensors
datasets
opencv-python
Pillow
numpy
einops
scipy
matplotlib
tensorboard
tqdm
lmms-eval
```

### 4.2 安装方式

```bash
cd /mingli01/project/xiyongkai/qwen3vl-seg
source activate segment
pip install -r requirements.in
pip freeze > requirements.lock
```

如果服务器存在内部 PyPI 镜像，使用镜像地址；否则使用默认源。安装后必须生成并提交 `requirements.lock`。

## 5. Slurm 模板

### 5.1 调试作业模板

```bash
#!/bin/bash
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=qvlseg-smoke
#SBATCH --output=/mingli01/project/xiyongkai/qwen3vl-seg/logs/slurm-%j.out

module load conda/3
module load cuda/13.0
source activate segment

cd /mingli01/project/xiyongkai/qwen3vl-seg
export DATA_ROOT=/file_storage01/home/mingli/data/xyk

torchrun --nproc_per_node=4 \
  -m qwen3vl_seg.train.training_loop \
  --config configs/train/stage1_small.yaml \
  --smoke
```

### 5.2 提交命令

```bash
sbatch scripts/launch_gpu.sh scripts/slurm/smoke_env.sh
squeue -u mingli
sacct -j <jobid> --format=JobID,State,ExitCode
```

## 6. 环境验收

### 6.1 登录节点

```bash
source activate segment
python -c "import torch, transformers, peft, accelerate; print(torch.__version__)"
```

### 6.2 4 卡计算节点

在 Slurm 作业中执行：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

预期：`True 4`。

### 6.3 DeepSpeed 验收

- 4 卡 DDP 启动成功。
- rank 0/1/2/3 输出一致。
- 简单张量同步后梯度一致。
- 峰值显存、吞吐、超时记录完整。

## 7. 验收标准

- `segment` 环境可激活。
- `requirements.lock` 存在且与 `requirements.in` 对应。
- `torchrun --nproc_per_node=4` 运行 4 卡 DDP 无错误。
- `torch.cuda.device_count() == 4`。
- Slurm 日志、退出码、资源使用记录完整。

## 8. 假设与失败处理

- 如果 CUDA 13.0 与 torch 2.4.0+cu121 出现兼容问题，记录并调整 torch 版本，不静默跳过。
- 如果 `/persist_data` 空间不足，所有缓存写入 `$DATA_ROOT`。
- 如果无法联网安装依赖，先报告，不伪造环境。
- 代码中的绝对路径只能来自 YAML 配置；环境变量仅作为覆盖，避免多用户运行不一致。
