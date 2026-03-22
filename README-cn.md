

# 情感多智能体系统

一个结合FinBERT（感知）和大语言模型（推理）的金融情感分析多智能体系统。

## 系统架构

该系统采用三智能体架构：

### 智能体A：感知智能体（FinBERT）
- 从金融文本中提取初始情感
- 使用FinBERT模型（ProsusAI/finbert）
- 提供情感评分：积极、中性、消极

### 智能体B：推理智能体（大语言模型）
- 执行推理和基于上下文的推断
- 使用Claude（Anthropic API）
- 提供分析、置信度和关键因素

### 智能体C：协调智能体
- 协调多智能体工作流程
- 整合两个智能体的输出
- 做出最终情感评估

## 安装指南

1. 克隆代码仓库：
```bash
git clone <仓库地址>
cd emotion_multiagent
```

2. 创建虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Windows系统: venv\Scripts\activate
```

3. 安装依赖包：
```bash
pip install -r requirements.txt
```

4. 设置环境变量：
```bash
cp .env.example .env
# 编辑.env文件，添加您的ANTHROPIC_API_KEY
```

## 项目结构

```
emotion_multiagent/
├── data/
│   ├── raw/                      # 原始数据文件
│   ├── processed/                # 处理后数据
│   └── external/                 # 外部词典/模型
├── src/
│   ├── agents/                   # 多智能体模块
│   │   ├── base_agent.py         # 智能体基类
│   │   ├── agent_a_perception.py # FinBERT智能体
│   │   ├── agent_b_inference.py  # 大语言模型智能体
│   │   ├── agent_c_coordinator.py # 协调智能体
│   │   └── prompts.py            # 提示词模板
│   ├── data/                     # 数据处理
│   │   ├── collector.py          # 数据采集
│   │   ├── preprocessor.py       # 数据预处理
│   │   └── stock_mapper.py       # 新闻-股票映射
│   ├── features/                 # 特征工程
│   │   └── sentiment_features.py # 情感特征
│   ├── evaluation/               # 评估指标
│   │   └── metrics.py            # 指标计算器
│   └── utils/                    # 工具函数
│       ├── config.py             # 配置管理
│       ├── logger.py             # 日志记录
│       └── retry.py              # 重试机制
├── config/
│   └── config.yaml               # 配置文件
├── outputs/
│   ├── features/                 # 生成的特征
│   └── logs/                     # 日志文件
├── notebooks/                    # Jupyter笔记本
├── tests/                        # 测试文件
├── main.py                       # 主程序入口
├── requirements.txt              # 依赖包列表
└── README.md                     # 本文件
```

## 使用方法

### 交互模式

在交互模式下运行系统，进行单条文本分析：

```bash
python main.py --mode interactive
```

### 流水线模式

对CSV文件进行批量处理：

```bash
python main.py --mode pipeline --input data/raw/financial_news.csv
```

### 自定义配置

使用自定义配置文件：

```bash
python main.py --config config/custom_config.yaml
```

## 配置说明

编辑 `config/config.yaml` 以自定义设置：

- **模型设置**：FinBERT模型、大语言模型选择
- **数据处理**：文本长度限制、数据集划分比例
- **特征工程**：滚动窗口、动量周期
- **输出设置**：日志和特征文件目录

## 功能特性

### 情感特征
- 每日情感评分
- 滚动平均值（3、7、14、30天）
- 动量指标
- 波动率指标
- 极端情感标记

### 评估指标
- 准确率、精确率、召回率、F1分数
- 混淆矩阵
- 智能体间一致性
- Cohen's Kappa系数

## 示例输出

```
处理记录 1/10
------------------------------------------------------------
分析结果
------------------------------------------------------------
文本: 苹果公司发布强劲季度财报，股价飙升5%

最终评估：
  情感倾向： 积极
  置信度： 0.875
  智能体一致性： 一致

感知层（FinBERT）：
  情感倾向：积极
  置信度： 0.985

推理层（大语言模型）：
  推理过程：强劲的财报表现和股价飙升表明积极情绪...
  情感倾向：积极
  影响因素：盈利增长、股价表现、公司实力
```

## 开发指南

### 运行测试

```bash
pytest tests/
```

### Jupyter笔记本

探索数据和原型开发智能体：

```bash
jupyter notebook notebooks/
```

## 环境要求

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.30+
- Anthropic API密钥

