# Data Engineering

**准备数据、构建学习分布，把含义明确的模型输入交给训练与研究。**

[English](README.md) · [数据消费](docs/consumption.md) · [序列打包](docs/packing.md) · [研究方法](docs/research.md) · [接口与安装](docs/integrations.md)

Data Engineering 是覆盖 AI 数据生命周期的独立研究环境，服务于预训练、后训练、检索、Agent、多模态系统与科学研究。数据保留领域结构，来源整理、分布选择和模型输入转换各有明确职责。数据方案的价值最终体现为：在确定的预算下，它如何改变使用这些数据的系统。

## 运行从数据到训练输入的工作流

需要 Python 3.11 或更新版本。Python 核心仅使用标准库。

```sh
git clone https://github.com/EnvFoundry/Data-Engineering.git
cd Data-Engineering
python -m pip install -e .
python examples/pack_training.py --output runs/packed-training
```

独立示例从人工编写的 token 记录开始，构建数据版本、采样分布和因果预测序列，再演示按训练进程分配数据及恢复读取位置。它解释真实的数据处理与目标对齐，不包含模型训练结果。

| 操作 | 提供的能力 |
| --- | --- |
| 准备 | 来源版本、分区、样本组关系、重复数据处理与转换记录。 |
| 组合 | 来源权重、记录或已有 token 数预算、索引分片，以及实际组成统计。 |
| 打包 | 固定长度的因果预测对、目标损失掩码、文档片段、位置与来源。 |
| 读取 | Python 与可选 Rust 读取器，按 rank 分配数据并恢复游标。 |
| 查看 | 有界的记录分页，以及 CLI/MCP 中按主题读取的研究指导。 |

```sh
ai-data profile examples/records.jsonl
ai-data build examples/local.recipe.json --output runs/source
python examples/compose_training.py --output runs/mixture-example
ai-data samples runs/mixture-example/mixture --limit 3
```

`python -m ai_data` 与 `ai-data` 等价。[数据契约](docs/contracts.md)允许保留偏好对、交互轨迹、图像与音频引用、传感器序列和表格。领域解码与模型专用分词由相应的数据处理步骤完成。

## 保持训练目标的含义

打包使用指定 tokenizer 已经产生的 token ID，并记录名称和版本。`input_ids[i]` 对应的预测目标是 `target_ids[i]`，损失掩码决定哪些目标参与优化。目标已经与 logits 对齐；对于内部自动移动标签的模型，使用[序列打包说明](docs/packing.md)中的显式适配方式。

片段信息保留文档边界，支持各文档独立的因果注意力。长记录切块时保留所有文档内相邻预测对；填充、目标掩码和 rank 尾部处理均有明确含义。读取游标应与模型和优化器状态一起保存。

```python
from ai_data import MixtureReader

with MixtureReader("runs/mixture-example/mixture", rank=1, world_size=4) as reader:
    record = next(reader)["record"]
    cursor = reader.state_dict()
```

[数据消费](docs/consumption.md)解释来源耗尽、加权与 rank 分配；[原生读取](docs/native-reader.md)解释可选 Rust 程序及兼容的游标格式。完整工作流也可以仅使用 Python。

## 接入 Harness 与 Agent

```sh
python -m pip install -e '.[mcp]'
ai-data capabilities
ai-data connect --root /absolute/path/to/data-workspace --output data-engineering.local.json
```

Python、CLI 和 MCP 共用实现。Agent 先获取统计与少量样本，再按需读取 payload 或 token。安装后的方法资源与提示词覆盖分布设计、训练打包和结果解释；仓库提供相应 Skill 与插件配置。[接口指南](docs/integrations.md)说明安装与扩展方式。

[TypeScript SDK](sdk/typescript/README.md)通过类型化方法启动已安装的 Python 包，支持取消、进程错误和有界响应。[Rust 读取器](docs/native-reader.md)则沿字节索引直接输出记录，供需要原生进程的消费者使用。

## 研究与扩展

来源配比、合成数据选择、任务难度、经验复用、多模态对齐和反馈更新，都应连接到具体的训练或使用结果。分别记录数据组成、系统吞吐与学习效果。[研究方法](docs/research.md)和[公开前沿工作](docs/frontier-workloads.md)给出可推进的方向。

连接器通过 `ai_data.connectors` 扩展；内置规范 JSONL 与 Gym 轨迹导入。可选的任务与经验工作流见[接口指南](docs/integrations.md)，默认运行不依赖其他 EnvFoundry 仓库。

目前来源准备仍将记录保留在内存；组合、打包与读取使用磁盘分片。[实现状态](docs/roadmap.md)区分已经可用的操作与后续扩展。代码位于 `src/ai_data/`、`rust/` 与 `sdk/typescript/`，可执行流程位于 `examples/`。贡献前阅读 [AGENTS.md](AGENTS.md)。MIT 许可证。
