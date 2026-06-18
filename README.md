# SmartRepo

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](#license)
[![Tests](https://img.shields.io/badge/Tests-18%20passed-brightgreen?style=flat)](#测试)
[![Models](https://img.shields.io/badge/Models-Claude_%7C_OpenAI_%7C_DeepSeek-blue?style=flat)](#配置-api-key)

面向代码仓库的本地多模型 AI 智能体。给它一句话任务,它自己读文件、搜索代码、执行命令、编辑文件——像一个系着安全带的程序员。本地运行,源码不离开你的机器。

支持 Claude、OpenAI、DeepSeek 三种 API,任选其一。

## 安装

需 Python 3.10+。

```bash
# 1. 克隆仓库
git clone https://github.com/<your-username>/SmartRepo.git
cd SmartRepo

# 2. 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell)
# source .venv/bin/activate     # macOS / Linux

# 3. 安装项目(可编辑模式,自动装依赖)
pip install -e .
```

开发依赖(含 pytest):`pip install -e ".[dev]"`。

## 配置 API Key

三选一(或多个)。在项目根建 `.env` 文件(已被 `.gitignore` 忽略,不会提交):

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...      # Claude
OPENAI_API_KEY=sk-...             # OpenAI
DEEPSEEK_API_KEY=sk-...           # DeepSeek
```

或临时用环境变量:

```bash
export DEEPSEEK_API_KEY="sk-..."        # bash / git bash
# $env:DEEPSEEK_API_KEY="sk-..."        # PowerShell
```

## CLI 命令

```bash
# 运行新任务(核心命令)
smart-repo run "找出所有循环导入并修复"
smart-repo run -m deepseek-chat -p deepseek "重构 src/auth.py"   # 指定模型/提供商
smart-repo run -i "删除未使用的 import"                            # -i:高风险步骤交互确认

# 从检查点恢复中断的任务(Ctrl+C 中断后可继续)
smart-repo resume <会话ID>

# 列出所有已保存的会话
smart-repo list

# 查看某会话的检查点
smart-repo checkpoints <会话ID>

# 显示运行统计(工具数、缓存、审批)
smart-repo stats
```

`run` 常用参数:`-m 模型` · `-p claude|openai|deepseek` · `-w 工作区` · `-t 最大轮次` · `-i 交互审批` · `-s 自定义系统提示`。

## Python 调用

```python
import asyncio
from smart_repo import SmartRepo

async def main():
    # with 自动管理 SQLite 检查点连接
    with SmartRepo(workspace_dir="/path/to/repo",
                   provider="deepseek", model="deepseek-chat") as sr:
        session = await sr.run("重构数据库连接层")
        print(session.state.value, session.turn_count, session.total_tokens_used)
        print(session.get_last_message().content)

asyncio.run(main())
```

恢复中断的任务:

```python
with SmartRepo() as sr:
    sessions = sr.list_sessions()
    session = await sr.resume(session_id=sessions[0])
```

## 原理(简)

主循环:**模型调用 → 工具执行 → 检查点**,循环到任务完成或达轮次上限。

- **断点恢复** — 每次工具调用存 SQLite 检查点,`Ctrl+C` 后 `resume` 从断点继续
- **上下文治理** — 对话过长时按"截断 → 摘要 → 丢弃 → 激进截断"逐层压缩,系统提示永不丢
- **安全** — 路径隔离 → 参数校验 → 分级审批 → 输出脱敏(10 类密钥)
- **多模型** — Claude/OpenAI/DeepSeek 统一一套接口,新增 provider 继承 `BaseProvider` 即可

## 测试

```bash
pytest smart_repo/benchmarks/      # 18 项,不依赖真实 API Key
```

## License

MIT。
