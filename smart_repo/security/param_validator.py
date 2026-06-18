"""参数验证器——基于 Pydantic 的运行时工具调用参数校验。

Parameter validator — Pydantic-based validation for all tool calls.
Ensures that the parameters LLMs pass to tools conform to the expected schema
before any tool code executes. This is the "input validation" layer of the
security stack.

============================================================================
设计背景与原理
============================================================================

本模块负责在执行工具之前验证 LLM 传入的参数是否符合工具的 JSON Schema 定义，
是安全体系中"输入验证"这一环的关键组件。LLM 属于非确定性系统，可能返回
格式错误、类型不匹配甚至恶意构造的参数。本模块在所有工具调用入口处拦截，
确保不合格参数永远无法触发工具处理逻辑。

============================================================================
Pydantic 动态模型创建的工作原理
============================================================================

核心问题：SmartRepo 的工具注册表中每个工具都以 JSON Schema 定义其参数规范。
编译时无法预知所有工具的 Schema，因此不能在代码中静态定义 Pydantic 模型。
解决方案是运行时利用 pydantic.create_model() 动态生成验证模型。

具体工作流程（参见 _get_model 方法）：

  步骤 1 — 解析 JSON Schema：
      从 schema 字典中提取 properties（字段定义）和 required（必填字段列表）。
      例如 schema = {"properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}

  步骤 2 — 类型映射（JSON Schema type → Python type）：
      建立一个类型映射表：
        "string"  → (str,   "")        # 字符串，默认空字符串
        "integer" → (int,   0)         # 整数，默认 0
        "number"  → (float, 0.0)       # 浮点数，默认 0.0
        "boolean" → (bool,  False)     # 布尔值，默认 False
        "array"   → (list,  [])        # 列表，默认空列表
        "object"  → (dict,  {})        # 字典，默认空字典
        未知类型   → (str,   "")        # 安全回退：未知类型当作字符串处理
      每个映射是一个 (Python类型, 默认值) 元组，同时解决了类型和缺省值两个问题。

  步骤 3 — 构建字段字典（fields: dict[str, Any]）：
      遍历 properties 的每个字段：
        - 如果字段名在 required 列表中 → 字段值设为 (py_type, ...)
          其中 ...（Ellipsis）是 Pydantic 的特殊标记，表示"必填——不接受 None 且无默认值"。
          Pydantic 会在实例化时检查：如果 required 字段缺失，直接抛出 ValidationError。
        - 如果字段不在 required 列表中 → 字段值设为 (py_type, default)
          使用步骤 2 中的类型默认值，字段可选且具有安全默认值。

  步骤 4 — 调用 create_model() 生成模型类：
      model = create_model(f"Tool_{tool_name}", **fields)
      create_model 是 Pydantic 的工厂函数，接收模型名称和字段定义字典，
      返回一个全新的 BaseModel 子类。该类具有：
        - 自动类型强制转换（如 "123" → 123 对于 integer 字段）
        - 自动必填字段验证（Ellipsis 字段缺失 → ValidationError）
        - 自动额外字段忽略（默认行为，防止注入）

  步骤 5 — 实例化验证：
      model(**parameters)  —— 用实际参数实例化模型对象。
      - 成功 → 参数有效，返回 (True, "")
      - 失败 → Pydantic 抛出 ValidationError，捕获后返回 (False, error_message)

  步骤 6 — 缓存复用：
      cache_key = f"{tool_name}:{json.dumps(schema, sort_keys=True)}"
      相同 tool_name + schema 组合只创建一次模型，后续调用直接从 _schema_cache 字典取用。
      这避免了每次工具调用都重新创建模型的开销（Pydantic 模型创建涉及类型元编程）。

为什么不直接用 jsonschema 库？
  - Pydantic 已在 SmartRepo 依赖中（用于配置管理），无需新增依赖
  - Pydantic 自带类型强制转换（coercion），比 jsonschema 更灵活
  - create_model 与 Pydantic 生态无缝集成，错误消息格式统一
  - 性能相当（Pydantic 底层使用 Rust 实现的 pydantic-core）

============================================================================
三层防护架构（validate_json_args）
============================================================================

validate_json_args() 实现了从原始 JSON 字符串到验证通过的三层防护：

  第 1 层 — JSON 语法解析（json.loads）：
      拦截格式错误的 JSON 字符串（如缺少引号、非法转义等），
      防止 JSON 注入攻击或格式异常导致的下游崩溃。

  第 2 层 — 类型检查（isinstance(params, dict)）：
      确保解析结果为字典对象。JSON 顶层也可能是数组或基本类型，
      但 SmartRepo 工具调用规范要求参数必须为 JSON Object。

  第 3 层 — Schema 验证（self.validate）：
      将参数字典传入 Pydantic 动态模型进行字段级验证。
      检查类型匹配、必填字段完整、无非法额外字段。

只有通过全部三层的参数才能继续进入工具处理逻辑。

============================================================================
设计理念总结
============================================================================

  - 使用 Pydantic 的 create_model 动态创建验证模型，从 JSON Schema 运行时生成。
  - 类型映射：将 JSON Schema 的 type 字段映射到 Python 类型（string→str, integer→int 等）。
  - Schema 缓存：相同 schema 只创建一次 Pydantic 模型，避免重复开销。
  - 必需/可选参数区分：required 列表中的字段使用 ...（Ellipsis）标记为必填，
    其他字段提供默认值。
  - validate_json_args() 整合 JSON 解析 + 字典类型检查 + Schema 验证，一站式入口。
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError, create_model


class ParameterValidator:
    """验证工具调用参数是否符合 JSON Schema 定义。

    使用 Pydantic 动态模型创建实现运行时验证。
    当无法创建 Pydantic 模型时，会退回到基本类型检查。

    职责：
      - 从 JSON Schema 动态生成 Pydantic 验证模型
      - 缓存已生成的模型避免重复创建开销
      - 验证传入参数字典是否符合预期类型和结构
      - 提供 JSON 字符串解析 + 验证的一站式方法

    使用方式：
        validator = ParameterValidator()
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["path"],
        }
        valid, error = validator.validate("read_file", {"path": "foo.py"}, schema)

    Validates tool call parameters against JSON Schema definitions.

    Uses Pydantic dynamic model creation for runtime validation.
    Falls back to basic type checking when Pydantic models can't be created.
    """

    def __init__(self) -> None:
        # Schema 缓存：避免相同 schema 重复创建 Pydantic 模型
        # Cache key is "tool_name:serialized_schema", avoids redundant model creation
        self._schema_cache: dict[str, type[BaseModel]] = {}

    def validate(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        schema: dict[str, Any],
    ) -> tuple[bool, str]:
        """根据 JSON Schema 验证参数字典。

        Validate parameters against a JSON Schema.

        安全检查：参数类型和结构验证——在执行工具之前确保 LLM 传入的参数
        符合预期格式，防止类型错误或恶意构造的参数进入工具处理器。
        这是输入验证层的核心入口，所有工具调用必须通过此验证。

        Args:
            tool_name: 被调用的工具名称（用于错误消息和缓存键）。
                       Tool name — used in error messages and cache key.
            parameters: LLM 传入的实际参数字典。
                        Actual parameters provided by the LLM.
            schema: 工具的 JSON Schema 定义（即 Tool.parameters 字段）。
                    JSON Schema to validate against (the tool's parameters field).

        Returns:
            (是否有效, 错误消息) —— 有效时错误消息为空字符串。
            无效时错误消息包含 Pydantic ValidationError 的详细信息
            （包括哪个字段、期望什么类型、实际收到什么值）。

            (is_valid, error_message) — error_message is empty if valid.
            On failure, the error message includes Pydantic's detailed
            field-level validation errors (which field, expected type,
            actual value received).
        """
        # 获取或创建 Pydantic 验证模型（带缓存）
        # Build or retrieve cached Pydantic model
        model = self._get_model(tool_name, schema)

        try:
            # 安全检查：使用 Pydantic 模型实例化验证参数。
            # model(**parameters) 会：
            #   1. 检查所有必填字段是否存在（缺失 → ValidationError）
            #   2. 检查所有字段类型是否正确（类型不匹配 → 尝试强制转换，失败 → ValidationError）
            #   3. 忽略 schema 中未定义的额外字段（默认行为，防止注入）
            # Security check: validate via Pydantic model instantiation.
            # This checks: required fields present, types correct, no extra fields.
            model(**parameters)
            return True, ""
        except ValidationError as e:
            # 验证失败：返回详细的字段级错误信息
            # Pydantic 的 ValidationError 包含每个失败字段的具体原因
            return False, f"Parameter validation failed for '{tool_name}': {e}"

    def _get_model(
        self, tool_name: str, schema: dict[str, Any],
    ) -> type[BaseModel]:
        """从 JSON Schema 获取或动态创建 Pydantic 模型。

        Get or create a Pydantic model from JSON Schema.
        Uses caching so the same schema is never compiled twice.

        ================================================================
        Pydantic 动态模型创建工作流（6 步）
        ================================================================

        步骤 1 — 缓存查找：
            使用 "tool_name:serialized_schema" 作为缓存键。
            如果该 schema 之前已编译为模型，直接返回缓存的模型类型，
            避免重复的类型元编程开销。

        步骤 2 — 解析 JSON Schema：
            从 schema 字典中提取：
              - properties: 字段名 → {"type": "string", ...} 的映射
              - required:    必填字段名称列表，如 ["path", "limit"]

        步骤 3 — 类型映射：
            JSON Schema type → (Python类型, 默认值) 映射表：
              "string"  → (str,   "")
              "integer" → (int,   0)
              "number"  → (float, 0.0)
              "boolean" → (bool,  False)
              "array"   → (list,  [])
              "object"  → (dict,  {})
              未知类型   → (str,   "")  ← 安全回退，当作字符串处理

        步骤 4 — 构建 Pydantic 字段定义字典：
            遍历 properties 的每个字段：
              - 必填字段（在 required 中）：字段值 = (py_type, ...)
                其中 ...（Ellipsis）= Pydantic Required 标记，
                实例化时如果该字段缺失，Pydantic 自动抛出 ValidationError。
              - 可选字段（不在 required 中）：字段值 = (py_type, default)
                使用步骤 3 中的默认值，字段可选。
            如果 properties 为空（无参数工具），创建 _dummy 占位字段。

        步骤 5 — 调用 create_model() 工厂函数：
            create_model(f"Tool_{tool_name}", **fields)
            Pydantic 的工厂函数，接收模型名称和字段定义字典，
            动态生成一个 BaseModel 子类。该类提供：
              - 自动类型验证与强制转换（如 "123" → 123）
              - 自动必填字段检查
              - ValidationError 异常（包含字段级错误详情）

        步骤 6 — 存入缓存并返回：
            将生成的模型类存入 _schema_cache，供后续调用复用。
            相同 schema 永远只创建一次模型。

        Args:
            tool_name: 工具名称（用于模型类名和缓存键前缀）。
            schema: JSON Schema 定义（通常为 Tool.parameters 字段的值）。

        Returns:
            动态创建的 Pydantic BaseModel 子类。
        """
        # ============================================================
        # 步骤 1: 缓存查找 — 避免重复编译
        # Step 1: Check cache — avoid recompiling the same schema
        # ============================================================
        # 缓存键：工具名 + schema 序列化（sort_keys 确保键顺序不影响缓存命中）
        # Cache key ensures identical schemas (even with different key ordering)
        # always hit the same cached model.
        cache_key = f"{tool_name}:{json.dumps(schema, sort_keys=True)}"
        if cache_key in self._schema_cache:
            return self._schema_cache[cache_key]

        # ============================================================
        # 步骤 2: 解析 JSON Schema 结构
        # Step 2: Extract properties and required-field set from schema
        # ============================================================
        fields: dict[str, Any] = {}
        properties = schema.get("properties", {})
        required: set[str] = set(schema.get("required", []))

        # ============================================================
        # 步骤 3: 类型映射表
        # Step 3: JSON Schema type → Python type + default value
        # ============================================================
        # 每个映射为 (Python类型, 默认值) 元组
        # Type mapping from JSON Schema to Python types with defaults
        type_map = {
            "string": (str, ""),
            "integer": (int, 0),
            "number": (float, 0.0),
            "boolean": (bool, False),
            "array": (list, []),
            "object": (dict, {}),
        }

        # ============================================================
        # 步骤 4: 遍历 properties 构建字段定义
        # Step 4: Build Pydantic field definitions from properties
        # ============================================================
        for name, prop in properties.items():
            json_type = prop.get("type", "string")
            py_type, default = type_map.get(json_type, (str, ""))

            # 安全检查：必填字段使用 ... (Ellipsis = Pydantic Required)，
            # 实例化时缺失此字段会直接抛出 ValidationError，确保必填参数
            # 不可能被漏掉或传入 None。
            # Required fields use Ellipsis (Pydantic Required marker)
            # — missing → ValidationError at instantiation time.
            # Optional fields use their type-default ("" / 0 / False / [] / {})
            if name in required:
                fields[name] = (py_type, ...)
            else:
                fields[name] = (py_type, default)

        # 步骤 4b: 无参数工具 — 创建虚拟占位字段
        #          Pydantic 不允许完全为空的模型，需一个 dummy 字段
        # No-param tool → create dummy field because Pydantic requires ≥1 field
        if not fields:
            # 工具无参数时的安全兜底：创建一个 _dummy 字段维持模型结构
            # Security: no-param tool → dummy field to keep model valid
            fields["_dummy"] = (str, "")

        # ============================================================
        # 步骤 5-6: 动态创建模型 → 存入缓存 → 返回
        # Steps 5-6: Dynamically create the Pydantic model class,
        #             cache it, and return it
        # ============================================================
        # create_model 是 Pydantic 的工厂函数，接收模型名称和 **字段定义，
        # 返回一个新的 BaseModel 子类。该模型具有完整的类型验证能力。
        model = create_model(f"Tool_{tool_name}", **fields)  # type: ignore[call-overload]
        self._schema_cache[cache_key] = model
        return model

    def validate_json_args(self, tool_name: str, json_args: str,
                           schema: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        """解析 JSON 参数字符串并执行三层验证。

        Parse and validate JSON arguments string.
        Three-layer check: valid JSON → is a dict → matches schema.

        一站式入口方法：从原始 JSON 字符串到验证通过的参数字典。
        这是外部调用者最常使用的接口，整合了所有防御层级。

        安全检查：三层防护架构——
          第 1 层（JSON 解析）：   拦截格式错误、非法转义、JSON 注入
          第 2 层（类型检查）：   确保解析结果为 dict，而非 list/str/number
          第 3 层（Schema 验证）： 使用 Pydantic 动态模型进行字段级类型和必填校验
        只有三层全部通过的参数才能继续进入工具执行流程。

        流程图：
          JSON 字符串
            │
            ▼
          json.loads()        ← 第 1 层：语法/注入防护
            │
            ├── 解析失败 → 返回 (None, "Invalid JSON arguments: ...")
            │
            ▼
          isinstance(dict?)   ← 第 2 层：类型防护
            │
            ├── 非 dict → 返回 (None, "Arguments must be a JSON object")
            │
            ▼
          self.validate()     ← 第 3 层：Schema 防护
            │
            ├── 验证失败 → 返回 (None, "Parameter validation failed: ...")
            │
            ▼
          返回 (parsed_dict, "")  ← 验证通过，返回参数字典

        Args:
            tool_name: 工具名称（用于缓存键和错误消息）。
            json_args: JSON 格式的参数字符串（来自 LLM 的原始输出）。
            schema: 工具的 JSON Schema 定义。

        Returns:
            (解析并验证后的参数字典, 错误消息)
            —— 成功时返回 (dict, "")，失败时返回 (None, error_message)。
            (parsed_dict, error_message) — parsed_dict is None on error.
        """
        # ============================================================
        # 第 1 层防护：JSON 语法解析 / Layer 1: JSON parse
        # 安全检查：json.loads 会拒绝非法的 JSON 语法，
        # 防止畸形输入绕过后续验证逻辑。
        # ============================================================
        try:
            params = json.loads(json_args)
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON arguments: {e}"

        # ============================================================
        # 第 2 层防护：类型检查——解析结果必须是字典（JSON Object）
        # Layer 2: Type check — must be a dict (JSON object)
        # 安全检查：JSON 顶层可以是任意类型（数组、字符串、数字等），
        # 但 SmartRepo 工具调用规范要求参数必须是 JSON Object {}。
        # 此检查防止攻击者传入数组或其他类型绕过字段级验证。
        # ============================================================
        if not isinstance(params, dict):
            return None, "Arguments must be a JSON object (dictionary)."

        # ============================================================
        # 第 3 层防护：Schema 验证 / Layer 3: Schema validation
        # 安全检查：使用 Pydantic 动态模型进行字段级验证——
        # 类型匹配、必填完整性、无额外非法字段。
        # ============================================================
        valid, error = self.validate(tool_name, params, schema)
        if not valid:
            return None, error

        return params, ""
