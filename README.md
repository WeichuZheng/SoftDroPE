# SoftDroPE: 长上下文位置编码扩展研究

## 项目简介

本项目是上海交通大学CS7352课程大作业，实现了SoftDroPE——一种结合DroPE和CoPE的混合后训练位置编码校准方法，用于大幅扩展大语言模型的上下文长度。

**核心思想**：
1. **Stage 1 (DroPE)**: 移除RoPE位置编码，进行短时间重校准，消除位置归纳偏置
2. **Stage 2 (CoPE)**: 注入软截断RoPE (CoPE)，实现平滑的频率正则化

**目标**: 实现10-100倍的上下文扩展 (最高支持128k tokens)

## 项目结构

```
大作业/
├── README.md                    # 本文件
├── requirements.txt             # Python依赖
├── src/
│   └── position/
│       ├── rope.py             # 标准RoPE实现
│       ├── cope.py             # CoPE (软截断RoPE)
│       ├── drope.py            # DroPE实现
│       ├── softdrope.py        # SoftDroPE (组合方法)
│       └── baselines.py        # 基线方法 (PI, NTK, YaRN)
├── ruler/
│   └── scripts/
│       ├── eval_hf.py          # RULER评测脚本 (HuggingFace)
│       ├── run_benchmark.py    # 完整基准测试脚本
│       └── monitor_gpu.py      # GPU监控脚本
├── scripts/
│   ├── download_models.py      # 模型下载脚本
│   └── monitor_gpu.py          # GPU监控脚本
├── data/
│   └── ruler/                  # RULER评测数据集
│       ├── L4096/              # 4096长度测试数据
│       └── L8192/              # 8192长度测试数据
└── results/
    └── evaluation/             # 实验结果
```

## 环境配置

### 1. 安装micromamba

```bash
# 安装micromamba (如果尚未安装)
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /data1/weichu/.local/bin
export PATH=/data1/weichu/.local/bin:$PATH
```

### 2. 创建环境并安装依赖

```bash
# 创建新环境
micromamba create -n softdrope python=3.10 -y

# 安装PyTorch (CUDA 12.1)
micromamba install -n softdrope pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 安装其他依赖
micromamba install -n softdrope transformers accelerate datasets tqdm scipy -c conda-forge -y

# 安装额外依赖
micromamba run -n softdrope pip install wonderwords nltk tenacity
```

## 使用方法

### 1. 位置编码模块测试

```bash
cd /data2/weichu/CS7352/大作业
micromamba activate softdrope

# 测试所有位置编码方法
python -c "
import sys
sys.path.insert(0, 'src')
import torch
from src.position.baselines import create_all_position_encoders

methods = ['rope', 'drope', 'cope', 'softdrope', 'pi', 'ntk', 'yarn']
for m in methods:
    encoder = create_all_position_encoders(m, dim=64)
    q = torch.randn(2, 4, 128, 64)
    k = torch.randn(2, 4, 128, 64)
    q_rot, k_rot = encoder(q, k)
    print(f'{m}: OK')
"
```

### 2. RULER基准评测

#### 2.1 生成测试数据

```bash
cd ruler/scripts
export CUDA_VISIBLE_DEVICES=0

# 生成不同长度的测试数据
micromamba run -n softdrope python data/prepare.py \
    --save_dir ../../data/ruler/L4096 \
    --benchmark synthetic \
    --task niah_single_1 \
    --tokenizer_path /data1/weichu/woshi/Qwen2.5-7B-Instruct \
    --tokenizer_type hf \
    --max_seq_length 4096 \
    --model_template_type base \
    --num_samples 10
```

#### 2.2 运行评测

```bash
cd /data2/weichu/CS7352/大作业
CUDA_VISIBLE_DEVICES=0 micromamba run -n softdrope python ruler/scripts/eval_hf.py \
    --model-path /data1/weichu/woshi/Qwen2.5-7B-Instruct \
    --data-dir data/ruler/L4096 \
    --task niah_single_1 \
    --device cuda:0 \
    --output results/evaluation/eval_L4096.json
```

### 3. GPU监控脚本

当GPU被占用时，可以使用监控脚本等待GPU空闲：

```bash
cd /data2/weichu/CS7352/大作业
micromamba run -n softdrope python scripts/monitor_gpu.py \
    --model-path /data1/weichu/woshi/Qwen2.5-7B-Instruct \
    --min-memory-gb 35 \
    --check-interval 60
```

## 实验结果

### 基线测试 (Qwen2.5-7B-Instruct)

| 序列长度 | 准确率 |
|---------|--------|
| 4096    | 100.00% |
| 8192    | 50.00%  |

**观察**: 模型在4096长度表现完美，但在8192长度下降到50%，说明需要位置编码扩展方法来改善长上下文性能。

## 项目时间线

| 周次 | 任务 | 负责人 |
|-----|------|--------|
| 9-10 | 环境配置，下载基础模型，实现DroPE重校准 | 郑伟初 |
| 11-12 | 实现CoPE，集成两阶段SoftDroPE管道 | 高岚迪 |
| 13-14 | 运行RULER评测，收集原始数据 | 谢吉利 |
| 15-16 | 频率分析，消融实验，准备展示 | 全体 |
| 17-18 | 撰写最终报告，整理代码仓库 | 全体 |

## 技术细节

### 位置编码方法

1. **RoPE** (Rotary Position Embedding): 标准旋转位置编码
2. **CoPE** (Clipped RoPE): 软截断RoPE，公式:
   ```
   Θ' = Θ × 0.5 × (1 + cos(Θ/Θcutoff × π))  for Θ < Θcutoff
   Θ' = Θ  otherwise
   ```
3. **DroPE**: 完全移除位置编码，通过重校准恢复
4. **PI** (Position Interpolation): 线性缩放位置索引
5. **NTK-aware**: NTK感知缩放
6. **YaRN**: 结合NTK和温度项的扩展方法

### 评估基准

使用 [NVIDIA RULER](https://github.com/NVIDIA/RULER) 基准测试，包含13个任务：
- NIAH (Needle In A Haystack)
- Variable Tracking
- Common Words Extraction
- Frequent Words Extraction
- QA tasks

## 注意事项

1. **GPU需求**: 至少需要35GB空闲GPU内存 (推荐RTX 4090)
2. **网络**: 部分依赖需要从HuggingFace下载，如网络不通可使用ModelScope
3. **当前模型**: 项目使用本地 `/data1/weichu/woshi/Qwen2.5-7B-Instruct` 模型

## 后续工作

- [ ] 实现完整的SoftDroPE训练管道
- [ ] 在GPT-2上测试
- [ ] 扩展到Llama-3-8B
- [ ] 实现PI、NTK、YaRN的模型修改
- [ ] 频率分析可视化

## 参考文献

1. Su et al. (2024). RoFormer: Enhanced transformer with rotary position embedding.
2. Chen et al. (2023). Extending context window of large language models via positional interpolation.
3. Peng et al. (2023). YaRN: Efficient context window extension of large language models.
4. Sakana AI (2025). DroPE: Dropping positional embeddings for zero-shot context extension.
5. Liu, Wu, & He (2026). CoPE: Clipped rotary position embedding for scalable length generalization.

---

**课程**: CS7352 深度生成模型编程实践  
**学期**: 2026 Spring  
**团队**: 郑伟初, 高岚迪, 谢吉利