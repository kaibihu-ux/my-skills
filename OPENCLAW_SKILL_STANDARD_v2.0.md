# OpenClaw Agent (Skill) 标准化开发规范

**Document Version:** v2.0 (Official Standard for OpenClaw Ecosystem)

---

## 核心开发流程（强制）

每个 Skill 开发必须经过以下流程：

```
用户需求 → 前置设计与流程规划(0) → 命名规范(1) → 工程结构(2) → 生命周期接口(3) → 编码规范(4) → 配置依赖(5) → I/O协议(6) → 测试规范(7) → 构建部署(8) → 文档规范(9) → Git版本管理(10)
```

---

## 0. 前置设计与流程规划（强制）

开发前必须由 OpenClaw 自动生成规划文档（存放于 `docs/plan/`）：

- `skill_design.md` — 设计规划文档（遵循附录C模板）
- `skill_process.mmd` — 功能流程图（mermaid格式）

**必须包含的流程节点：**
- 触发节点 → 参数校验节点 → 核心功能执行节点 → 分支节点（成功/失败）→ 结束节点/异常节点

---

## 1. 命名规范（强制）

**Skill 命名：** `{business_domain}_{function}_skill`（全小写、下划线分隔）

**目录结构：**
```
skill-root/
├── config/          # 环境配置（dev/test/prod）
├── src/
│   ├── core/        # 业务逻辑
│   ├── api/         # 接口封装
│   ├── utils/       # 工具
│   └── constant.py  # 常量
├── test/
│   ├── unit/        # 单元测试
│   └── integration/ # 集成测试
├── docs/plan/       # OpenClaw 自动生成的规划文档
├── scripts/         # 构建/部署脚本
├── logs/            # 运行日志（gitignore）
├── openclaw.json    # 技能元数据
├── README.md
└── requirements.txt
```

---

## 3. 生命周期接口（强制）

```python
from openclaw import SkillBase

class StandardSkill(SkillBase):
    def __init__(self):
        super().__init__()

    def execute(self, params: dict) -> dict:
        # 【唯一执行入口】
        pass

    def pause(self):
        # 保存现场
        pass

    def resume(self):
        # 恢复执行
        pass

    def destroy(self):
        # 释放资源
        pass
```

---

## 4. 编码规范

- 禁用 `print()`，使用 `self.logger`
- 日志分级：debug/info/warn/error
- 异常必须携带堆栈信息
- 圈复杂度 ≤ 10
- 无硬编码敏感信息

---

## 6. I/O 协议（强制）

**输入：**
```json
{ "request_id": "uuid-string", "params": {} }
```

**输出：**
```json
{ "code": 0, "msg": "success", "data": {}, "request_id": "uuid-string" }
```

---

## 7. 测试规范（强制）

- 单元测试覆盖率 ≥ 80%
- 使用 `openclaw skill test` 运行测试
- 准入标准：无失败用例、覆盖率达标、性能达标

---

## 10. 版本与 Git 规范

**语义化版本：** 主版本.次版本.修订号

**分支规范：**
- `main` — 生产稳定版
- `develop` — 开发主干
- `feature/*` — 功能开发
- `bugfix/*` — 问题修复

---

## 交付校验清单

- ✅ 完成前置设计与流程绘制
- ✅ 规划文档评审通过并归档
- ✅ 命名符合规范
- ✅ 标准工程结构（含 docs/plan）
- ✅ 完整生命周期接口
- ✅ I/O 协议合规
- ✅ 单元测试覆盖率 ≥80%
- ✅ 无敏感信息硬编码
- ✅ 文档完整
- ✅ 构建部署无异常
- ✅ 通过框架安全扫描
