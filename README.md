# SmartRepo

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](#license)
[![Tests](https://img.shields.io/badge/Tests-18%20passed-brightgreen?style=flat)](#测试)

本地运行的代码仓库 AI agent。给一句话任务,它读文件、搜索、执行命令、改代码,直到完成。源码不离开本机,接 Claude / OpenAI / DeepSeek 任一 API。

## 安装

需 Python 3.10+。

```bash
git clone https://github.com/JACK-123666/smartrepo.git
cd smartrepo
python -m venv .venv
.venv\Scripts\activate          # Windows; macOS/Linux: source .venv/bin/activate
pip install -e .
```

开发依赖:`pip install -e ".[dev]"`。

## 配置

项目根建 `.env`(已 gitignore),填一个 API key:

```ini
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
```

不指定模型时,自动用已配 key 的 provider。

## 用法

```bash
smart-repo run "找出所有循环导入并修复"
smart-repo run -m deepseek-chat -p deepseek "重构 src/auth.py"
smart-repo run -i "删除未使用的 import"        # -i:高风险步骤交互确认
smart-repo resume <id>                          # 恢复中断的任务
smart-repo list                                 # 列出会话
smart-repo stats                                # 运行统计
```

`run` 参数:`-m 模型` `-p claude|openai|deepseek` `-w 工作区` `-t 最大轮次` `-i 交互审批` `-s 系统提示`。

Python:

```python
import asyncio
from smart_repo import SmartRepo

async def main():
    with SmartRepo(workspace_dir="/path/to/repo") as sr:
        session = await sr.run("重构数据库连接层")
        print(session.get_last_message().content)

asyncio.run(main())
```

## 原理

主循环:模型调用 → 工具执行 → 检查点,循环到完成或达轮次上限。

- 每次工具调用存 SQLite 检查点,Ctrl+C 后 `resume` 从断点继续
- 对话过长时按"截断 → 摘要 → 丢弃 → 激进截断"逐层压缩,系统提示永不丢
- 操作经路径隔离、参数校验、分级审批、输出脱敏四道防线
- 新增 provider 继承 `BaseProvider` 实现两个方法即可

## 测试

```bash
pytest smart_repo/benchmarks/      # 18 项,不需 API key
```

## License

MIT。
