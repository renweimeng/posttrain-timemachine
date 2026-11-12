# 测试与验收记录（v0.2.0）

日期：2026-11-03，Asia/Shanghai。范围：本地 v0.1/v0.2 产品流程、真实双进程 CPU L1 采集、有限 CPU 单步 checkpoint replay。
新版网站运行在 http://127.0.0.1:8765；服务版本、全部页面入口和 API 已进行 HTTP 检查。

## 实际执行结果

| 检查 | 结果与证据 |
|---|---|
| `PTM_BROWSER_CHANNEL=chrome HF_HUB_OFFLINE=1 ./scripts/check.sh` | 退出码 0；`output/completion-full-check.log` |
| Python | **81 passed**，0 failures，0 errors；`output/pytest-junit.xml` |
| 语句覆盖率 | **84.78%**，2162/2550；`output/coverage.json` |
| 浏览器 | **12 passed**；另加覆盖 10 个页面状态的 axe 检查；`output/completion-e2e.log` |
| 静态检查 | Ruff、mypy、ESLint、TypeScript strict 通过；OpenAPI 类型重新生成 |
| 生产前端 / Storybook | 构建通过；所有运行时资源均在本地 |
| 原生 L2 训练 | 真实 TRL 0.24.0：GRPO、BNPO、DAPO、Dr.GRPO 四种 loss；AdamW 与 SGD；5 组实际训练测试 |
| 真正参数更新 | 保存 native before/after 权重，独立执行 forward/backward/clip/optimizer；原始更新与 native 权重相符，修改奖励分支实际执行但不声称 native parity |
| 原生分布式训练 | 实际 2-rank CPU gloo GRPO；全局组完整，原生 Advantage 对齐；`output/distributed-trl/result.json` |
| 离线报告 | L1/L2 导出、哈希篡改检测、Token/Objective 外键、原始及修改后 loss 的离线重算通过；`output/l2-native-report.zip` |
| 安装包 | v0.2.0 wheel；33 个源码/静态文件逐字节核对，8 个前端资源无旧版本残留；`output/release-verification.json` |
| 独立安装 | 在**没有 Torch** 的新 venv 中，Demo、API、L2 loss 和离线报告重算通过；更新按钮依赖缺失时有明确降级；`output/wheel-smoke.json` |
| 默认 Python / 生产前端依赖扫描 | 0 个已知公告；`docs/benchmarks/pip-audit-runtime.json`、`docs/benchmarks/npm-audit.json` |

浏览器覆盖：异常→样本→奖励修改→人工确认契约→失败/通过回归→下载报告；实际训练 trace；
四个 Arena 原始规则失败与修补通过；Token 深链接、mask、精确表格和 objective；真实单步参数更新；
文件导入及契约重新审核；刷新/返回/草稿恢复/历史报告；手机布局、空数据、错误态、XSS 文本与断网资源限制。
自动化无障碍检查按 WCAG 2 A/AA 与 2.1 AA 执行，覆盖 Timeline、Runs、Arena、Contracts、Reports、
Analysis History、What-if Lab、Update Replay、L1 Inspector、L2 Inspector，未发现 serious/critical 项。
这不等于完整人工无障碍认证。

## 数值和失败路径

真实 DAPO fixture 的 NumPy/native loss 差约 **1.86e-9**，实际 replay/native 权重最大误差约 **5.53e-11**。
测试容差为 loss atol=2e-6、rtol=1e-5，原始更新最大权重误差 ≤2e-6；测量值不是跨硬件误差保证。

独立标量参考与 Hypothesis 覆盖 group 排序、epsilon、NaN/Inf、缺失奖励、不完整组、极端值；
L2 覆盖中文/Emoji/EOS、padding、mask=0、缺失 old/current/ref、policy identity、完整 rank/microbatch grid、
分母不一致、四种 loss 和禁用的额外 modifier。缺失输入不填 0，不推断梯度或未来质量。
Token surrogate 使用关联 objective 的裁剪阈值，缺失/歧义时明确标注仅为显示计算器默认值。
Arena 已将不同题目的 known/guard 划为独立四候选组；新增回归按原始断言通过→当前失败计算。
非二元 verifier grade 在卡片与契约选择器中都保留数值语义。

序列化失败不改变 Trainer reward 的原始返回对象。采集 drop/error 进入健康信息；未知覆盖率不伪装完整。
报告不包含权重、优化器状态和任意私有配置；显式 numeric capture 单独存放。
多轮轨迹导入/检索/导出与 full/off/hash_only 保留策略均有测试；Token IDs 保留，文本移除不是匿名化保证。

## 性能测量

环境：macOS-26.5.2-arm64-arm-64bit，Python 3.12.11，arm64，16 GiB RAM，
本机文件系统，SSD 型号未读取。查询/DSL 测量时未同时运行 Trainer。原始值见 `docs/benchmarks/benchmark.json`。

| 测量 | 本机结果 | 测量边界 |
|---|---|---|
| 100 万 metadata 的预热组检索 P95 | 2.32 ms | Parquet/DuckDB 微基准，不是全 API/UI 首屏 |
| 10 万 completion 的简单 DSL 重评分 P95 | 112.46 ms | 含组件字典，不含导入/落盘/外部 Judge |
| Demo 准备，3 个新进程 | 0.341s, 0.328s, 0.308s | 未清 OS 缓存，不含浏览器启动 |
| 元数据压缩比 | 19.12× | Arrow 字节 / Parquet，无 Token |
| 5000 Token Canvas 滚动 | 62.07 FPS | 本机 headless Chrome，30 帧采样；可访问表按 100 个分页 |
| 10% 整组 L1 采样开销 | 配对中位数 0.87% | 5 组交替顺序、每组 40 步的小 CPU 模型；原始波动较大 |

不能将这些数据用于宣称百万样本完整首屏 P95、任意 Judge 吞吐、生产 GPU 的 ≤3% 采集开销或所有设备 ≥50 FPS。
采集原始配对值保存在 `docs/benchmarks/capture-benchmark.json`；仅该 toy 实验的中位数达到设计目标。

## 审查与执行边界

本轮两轮 assumption sweep：第一轮找到 3 个未声明语义决策、1 个接口范围遗漏、1 个过时记录；
均已修正并新增测试。第二轮 undeclared=0、stale_rows=0，检查范围内 clean。
审查路由配置为 gpt-6-astra xhigh；子会话未独立暴露精确 serving model ID，实际仅确认 GPT-6 家族。
因此为 **same-family / provisional**，不属于跨系列独立验收。未声明假设审查记录留在维护者本地；本仓库保留公开测试与数值证据。

默认运行环境与可选旧版 TRL reference extra 分开。完整 reference 环境先前同日扫描保留
16 条公告记录、涉及 3 个可选包；未隐藏/抑制。详见 `docs/SECURITY.md` 与 `docs/benchmarks/pip-audit.json`。
更新回放明确启用后才会加载这些包；普通网站启动不导入 Torch。

实际验证平台是 macOS / Python 3.12 / Chrome。Linux/macOS、Python 3.11/3.12 的远端 CI 配置已提供，
未在远端执行。`docs/benchmarks/preflight.json` 显示 CUDA 不可用；**GPU retrain、GPU 吞吐和生产模型训练未验证**。
原生 distributed L2、通用 Agent/OpenRLHF hook、异步 policy-step 距离推断和长期因果训练结论不属于已验证能力。
显式导入的 recorded trajectory 不执行工具；policy label 的字符串差异不等于已知 step lag。

上游 TestClient 的 httpx 弃用提示与 Monaco 延迟加载 chunk 大小提示保留；均未导致本次检查失败。
全部本地 MUST rung 检查通过，无运行中的数值占位实现。公开功能范围详见 requirements.md；完整本机日志不随源码发布。
