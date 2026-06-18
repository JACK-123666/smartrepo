"""敏感数据清洗器——凭据检测与掩码。

Sensitive data sanitizer — secret detection and masking.
Acts as the last line of defence for output safety: any credentials or tokens
that leak into tool output (despite sandboxing) are detected and masked before
the output reaches the LLM or user.

============================================================================
设计背景与原理
============================================================================

本模块是 SmartRepo 安全体系中"输出防护"的最后一道防线。即使沙箱和审批
机制正常运行，工具执行结果中仍可能意外包含敏感信息（如 API Key、Token、
密码等）。本模块在输出返回给 LLM/用户之前，自动检测并掩码这些敏感数据。

在 LLM Agent 系统中，工具调用的输出在返回给 LLM 之前会流经本模块。
如果这些输出中包含 API Key、Token、密码等敏感信息，LLM 可能将其"记住"
并在后续交互中泄露。本模块在数据流管道中充当安全过滤器。

============================================================================
10 种敏感信息检测模式及脱敏策略
============================================================================

  ┌────┬──────────────────────────┬──────────────────────────────────────┐
  │ #  │ 检测类型 (name)            │ 检测说明 / 掩码策略                   │
  ├────┼──────────────────────────┼──────────────────────────────────────┤
  │  1 │ anthropic_api_key        │ Anthropic API Key                    │
  │    │                          │ 格式: sk-ant- 前缀 + 可选版本号        │
  │    │                          │       + 50+ 位字母数字                │
  │    │                          │ 掩码: "sk-ant-***" — 保留前缀标识     │
  ├────┼──────────────────────────┼──────────────────────────────────────┤
  │  2 │ openai_api_key           │ OpenAI API Key                       │
  │    │                          │ 格式: sk- 前缀 + 可选 proj-           │
  │    │                          │       + 30+ 位字母数字                │
  │    │                          │ 掩码: "sk-***" — 保留平台标识         │
  ├────┼──────────────────────────┼──────────────────────────────────────┤
  │  3 │ github_token             │ GitHub Personal Access Token         │
  │    │                          │ 格式: ghp/gho/ghu/ghs/ghr 前缀       │
  │    │                          │       + 下划线 + 36+ 位字母数字       │
  │    │                          │ 掩码: "ghp_***" — 保留 token 类型标识 │
  ├────┼──────────────────────────┼──────────────────────────────────────┤
  │  4 │ aws_access_key           │ AWS IAM Access Key ID                │
  │    │                          │ 格式: AKIA 前缀 + 16 位大写字母数字   │
  │    │                          │ 掩码: "AKIA***" — 保留云服务商标识     │
  ├────┼──────────────────────────┼──────────────────────────────────────┤
  │  5 │ aws_secret_key           │ AWS Secret Access Key                │
  │    │                          │ 格式: 40 位 Base64 字符串             │
  │    │                          │       + aws/secret/key 上下文关键词   │
  │    │                          │ 掩码: "***" — 完全掩码（高危）        │
  ├────┼──────────────────────────┼──────────────────────────────────────┤
  │  6 │ jwt_token                │ JSON Web Token (JWT)                 │
  │    │                          │ 格式: eyJ 开头三段式                  │
  │    │                          │       header.payload.signature       │
  │    │                          │ 掩码: "***.***.***" — 保留三段结构   │
  ├────┼──────────────────────────┼──────────────────────────────────────┤
  │  7 │ private_key_header       │ 私钥 PEM 文件头                       │
  │    │                          │ 格式: -----BEGIN (RSA|DSA|EC|        │
  │    │                          │       OPENSSH|PGP) PRIVATE KEY-----  │
  │    │                          │ 掩码: "***PRIVATE KEY***"            │
  ├────┼──────────────────────────┼──────────────────────────────────────┤
  │  8 │ generic_api_key          │ 通用 API Key / Secret Key            │
  │    │                          │ 格式: api_key= / apikey= /           │
  │    │                          │       secret_key= 后 20+ 位值        │
  │    │                          │ (大小写不敏感)                        │
  │    │                          │ 掩码: "***API_KEY***"                │
  ├────┼──────────────────────────┼──────────────────────────────────────┤
  │  9 │ password_in_connection   │ 数据库连接字符串中的密码               │
  │    │                          │ 格式: mysql://user:PASS@host         │
  │    │                          │       postgres://user:PASS@host      │
  │    │                          │       mongodb://user:PASS@host       │
  │    │                          │       redis://user:PASS@host         │
  │    │                          │ 掩码: "***" — 完全掩码（高危）        │
  ├────┼──────────────────────────┼──────────────────────────────────────┤
  │ 10 │ discord_token            │ Discord Bot Token                    │
  │    │                          │ 格式: M/N/O 开头三段式                │
  │    │                          │       ID.WebhookSig.TokenSig         │
  │    │                          │ 掩码: "***" — 完全掩码               │
  └────┴──────────────────────────┴──────────────────────────────────────┘

============================================================================
脱敏策略总结
============================================================================

  1. 保留前缀策略（模式 1-4, 7）：
     掩码保留信息类型的可识别前缀或格式结构（如 "sk-ant-***"、
     "AKIA***"），便于运维和调试时识别被脱敏的信息类别，
     同时确保原始凭证不可恢复。

  2. 结构保留策略（模式 6）：
     对于 JWT 这种三段式结构，掩码 "***.***.***" 保留点号分隔，
     让人一眼看出这是 JWT Token 的残留痕迹。

  3. 完全掩码策略（模式 5, 9, 10）：
     对于高危凭证（AWS Secret Key、数据库密码、Discord Token），
     使用 "*" 完全替代，不保留任何可识别特征。

  4. 匹配顺序：按 PATTERNS 列表顺序从上到下依次匹配替换。
     先匹配的模式优先，已被替换的内容不会被后续模式再次匹配。

  5. 零状态纯函数：所有方法为 @classmethod，无实例状态，线程安全。

============================================================================
三种使用模式
============================================================================

  1. sanitize()     → 输出净化——在数据返回给 LLM/用户前自动掩码
  2. detect()       → 安全审计——检测并报告敏感信息但不修改原文本
  3. has_secrets()  → 快速判定——短路逻辑的条件分支检查
  + sanitize_dict() → 深度净化——递归处理嵌套 JSON/字典结构
"""

from __future__ import annotations

import re
from typing import Any


class SensitiveDataSanitizer:
    """检测和掩码输出中的敏感信息。

    所有方法均为类方法，无需实例化。直接调用 SensitiveDataSanitizer.sanitize(text) 即可。

    覆盖的凭据类型：
      - API 密钥（Anthropic、OpenAI、AWS、GitHub、通用格式）
      - 文本中的密码和令牌
      - 私钥（SSH、PGP）
      - 带凭据的连接字符串
      - JWT 令牌
      - Discord Bot 令牌

    Detects and masks sensitive information in outputs.

    Patterns covered:
      - API keys (Anthropic, OpenAI, AWS, GitHub, generic)
      - Passwords and tokens in text
      - Private keys (SSH, PGP)
      - Connection strings with credentials
      - JWT tokens
    """

    # 敏感数据检测的正则表达式模式列表
    # 每个条目: (模式名称, 掩码模板/替换字符串, 编译后的正则表达式)
    # 安全检查：这些模式用于在输出返回给用户/LLM之前识别并掩码凭据
    #
    # Patterns for sensitive data detection
    # Each entry: (name, mask_template/placeholder, compiled regex)
    # Security check: These patterns identify credentials in output before
    #   it is returned to the user or LLM, preventing credential leakage.
    PATTERNS: list[tuple[str, str, re.Pattern]] = [
        # (name, mask_template, regex)

        # Anthropic API Key: sk-ant- 开头，约 50+ 字符
        # Anthropic API key pattern: sk-ant- prefix, ~50+ chars
        ("anthropic_api_key", "sk-ant-***",
         re.compile(r'sk-ant-(?:api\d{2}-)?[a-zA-Z0-9_-]{50,}')),

        # OpenAI API Key: sk- 开头（可能含 proj-），约 30+ 字符
        # OpenAI API key pattern: sk- prefix (possibly with proj-), ~30+ chars
        ("openai_api_key", "sk-***",
         re.compile(r'sk-(?:proj-)?[a-zA-Z0-9_-]{30,}')),

        # GitHub Token: ghp/gho/ghu/ghs/ghr_ 开头，36+ 字符
        # GitHub personal access token / OAuth / user-to-server / server-to-server
        ("github_token", "ghp_***",
         re.compile(r'(?:ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{36,}')),

        # AWS Access Key: AKIA 开头，16 位大写字母数字
        # AWS IAM access key ID pattern
        ("aws_access_key", "AKIA***",
         re.compile(r'AKIA[0-9A-Z]{16}')),

        # AWS Secret Key: 40 位 base64 字符，附近有 "aws" + "secret"/"key" 上下文
        # AWS secret access key — requires contextual match (aws + secret/key nearby)
        ("aws_secret_key", "***",
         re.compile(r'(?i)aws.{0,20}(?:secret|key).{0,5}[\'"]?([a-zA-Z0-9/+]{40})')),

        # JWT Token: eyJ 开头，三段 base64url 以点号分隔
        # JWT token: eyJ header prefix, three dot-separated segments
        ("jwt_token", "***.***.***",
         re.compile(r'eyJ[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,}')),

        # 私钥文件头: -----BEGIN RSA/DSA/EC/OPENSSH/PGP PRIVATE KEY-----
        # Private key PEM header — detects the start of an inline private key
        ("private_key_header", "***PRIVATE KEY***",
         re.compile(r'-----BEGIN (?:RSA|DSA|EC|OPENSSH|PGP) PRIVATE KEY-----')),

        # 通用 API Key: api_key= / apikey= / secret_key= 后跟 20+ 字符值
        # Generic API key: api_key / apikey / secret_key with 20+ char value
        ("generic_api_key", "***API_KEY***",
         re.compile(r'(?:api[_-]?key|apikey|secret[_-]?key)["\s:=]+([a-zA-Z0-9_-]{20,})',
                    re.IGNORECASE)),

        # 数据库连接字符串中的密码: mysql://user:password@host 格式
        # Database connection string password: scheme://user:PASSWORD@host
        ("password_in_connection", "***",
         re.compile(r'(?:mysql|postgres|mongodb|redis)://[^:]+:([^@]+)@')),

        # Discord Bot Token: M/N/O 开头，三段以点号分隔
        # Discord bot token pattern
        ("discord_token", "***",
         re.compile(r'[MNO][a-zA-Z\d_-]{23,25}\.[a-zA-Z\d_-]{6}\.[a-zA-Z\d_-]{27}')),
    ]

    @classmethod
    def sanitize(cls, text: str) -> str:
        """将检测到的敏感信息替换为掩码占位符。

        安全检查：输出净化——遍历所有敏感数据模式，
        将匹配到的凭据替换为无害的掩码字符串（如 "sk-***"）。
        这确保即使凭据意外出现在工具输出中，也不会泄露给 LLM 或用户。

        Args:
            text: 要净化的文本。

        Returns:
            敏感信息已被掩码的净化后文本。

        Replace detected secrets with masked placeholders.
        """
        sanitized = text
        for name, mask, pattern in cls.PATTERNS:
            # 安全检查：用掩码替换匹配到的凭据 / Replace matched secret with mask
            sanitized = pattern.sub(mask, sanitized)
        return sanitized

    @classmethod
    def detect(cls, text: str) -> list[dict[str, Any]]:
        """检测文本中的敏感信息（不进行掩码替换）。

        用于审计和安全告警场景——不修改原文本，只返回检测到的凭据类型和位置。

        安全检查：凭据检测——用于识别输出中是否泄露了敏感信息，
        但不会修改原始内容。

        Args:
            text: 要扫描的文本。

        Returns:
            检测结果列表，每项包含：
              - type: 凭据类型名称
              - match_preview: 匹配内容的前 40 个字符预览
              - position: 匹配在文本中的起始位置
              - length: 匹配内容的长度

        Detect sensitive information without masking.
        Returns structured findings for audit/alerting purposes.
        """
        findings = []
        for name, mask, pattern in cls.PATTERNS:
            for match in pattern.finditer(text):
                preview = match.group(0)[:40]
                findings.append({
                    "type": name,
                    "match_preview": preview + ("..." if len(match.group(0)) > 40 else ""),
                    "position": match.start(),
                    "length": len(match.group(0)),
                })
        return findings

    @classmethod
    def has_secrets(cls, text: str) -> bool:
        """快速检查文本中是否包含任何敏感信息。

        比 detect() 更轻量——找到第一个匹配即返回 True，
        适用于条件分支中的快速判定。

        安全检查：快速凭据扫描——用于在管道中决定是否需要进一步处理。

        Args:
            text: 要检查的文本。

        Returns:
            如果检测到至少一个敏感模式则返回 True。

        Quick check if text contains any secrets.
        Returns True on the first match — lightweight pre-flight check.
        """
        for _, _, pattern in cls.PATTERNS:
            # 安全检查：扫描敏感模式 / Security check: scan for any secret pattern
            if pattern.search(text):
                return True
        return False

    @classmethod
    def sanitize_dict(cls, data: dict[str, Any]) -> dict[str, Any]:
        """递归净化字典中的所有字符串值。

        安全检查：深度数据净化——遍历字典、列表的嵌套结构，
        对所有字符串值调用 sanitize()。适用于处理复杂的 JSON 响应、
        配置文件等嵌套数据结构。

        Args:
            data: 可能包含嵌套字典/列表的输入数据。

        Returns:
            所有字符串值已经过 sanitize() 处理的净化后字典。

        Recursively sanitize all string values in a dict (and nested structures).
        Handles nested dicts, lists, and mixed structures.
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                # 安全检查：净化字符串值 / Sanitize string values
                result[key] = cls.sanitize(value)
            elif isinstance(value, dict):
                # 递归处理嵌套字典 / Recurse into nested dicts
                result[key] = cls.sanitize_dict(value)
            elif isinstance(value, list):
                # 处理列表中的字符串和字典 / Handle list items (strings, dicts, or other)
                result[key] = [
                    cls.sanitize(item) if isinstance(item, str)
                    else cls.sanitize_dict(item) if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                # 非字符串/字典/列表类型，原样保留 / Non-string/dict/list — pass through
                result[key] = value
        return result
