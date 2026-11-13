# BluePixel ComfyUI 节点实现与验证计划

更新时间：2026-11-10

## 1. 目标与当前状态

第一版提供接近 Klein 原版 pipeline 使用方式的 `BluePixel Pipeline Loader` 和 `BluePixel Generate` 两个节点，保留当前 BluePixel 推理语义，并接入 ComfyUI 的模型加载、卸载、逐层低显存执行、进度与取消机制。

当前阶段为方案设计。已完成仓库源码、模型配置和 ComfyUI 上游相关接口的静态核查；尚未实现节点，尚未安装并运行 ComfyUI，尚未执行真实权重推理、数值对齐或 GPU 显存测试。本文中的通过标准是后续验收要求，不代表测试已通过。

状态约定：

- **已核查**：已读取源码或配置，有静态证据；不代表运行成功。
- **待实现**：目标代码尚未完成。
- **待验证**：尚未执行相应测试，无运行结论。
- **通过 / 失败 / 阻塞**：仅在实际执行后填写，必须附环境、命令、结果与证据。

## 2. 已核查的事实与风险

| 项目 | 静态核查结论 | 证据 | 运行验证状态 |
| --- | --- | --- | --- |
| 推理入口 | `FluxKleinPipeline` 和 `load_pipeline()` 位于单文件中 | `vins_infer.py` | 待验证 |
| 模型加载 | 组件直接 `.to(device)`，最终调用 `pipe.to(device)` | `load_pipeline()` | 尚未具备 ComfyUI 托管能力 |
| DiT 配置 | 128 输入通道、12288 文本维度、guidance embedding 关闭 | 本地 `huggingface/BluePixel-Turbo-clone/transformer/config.json` | 待真实权重验证 |
| DCAE 配置 | 64 倍空间压缩、128 latent 通道、scaling factor 0.4538 | pipeline 与本地 VAE 配置 | 待编解码验证 |
| 文本编码 | Qwen3 第 9、18、27 层拼接，默认长度 512 | `_get_qwen3_prompt_embeds()` | 待 tensor 对齐 |
| CFG | guidance > 1 时执行条件与无条件两个 forward | `FluxKleinPipeline.__call__()` | 待分支与结果验证 |
| 调度 | empirical / training μ，以及固定 shift 路径 | `configure_inference_scheduler()`、`compute_inference_mu()` | 待 sigma 与逐步更新对齐 |
| 尺寸处理 | 存在 64 倍尺寸搜索和按 128 配置的 image processor | `FluxKleinPipeline` | 待确认最终尺寸与二次取整行为 |
| 依赖版本 | requirements 使用 Diffusers 0.36、Transformers 4.57 系列；本地导出配置记录更新版本 | `requirements.txt` 与本地模型配置 | 兼容性未验证，不判定一定不兼容 |
| lowvram | ComfyUI 的部分加载涉及支持权重迁移的算子；仅套 ModelPatcher 不足以证明支持 | 上游 model_patcher / ops 源码 | 待实际部分加载验证 |

本地模型目录属于现有未跟踪内容，本计划不修改它们，也不将其视为已经验证的发布模型。P0 必须确定正式测试权重与 revision。

## 3. 第一版接口与范围

工作流：

```text
BluePixel Pipeline Loader ── BLUEPIXEL_PIPE ──┐
                                            ├─ BluePixel Generate ─ IMAGE ─ Save/Preview Image
Load Image ─────────────────────── IMAGE ────┘
```

### Loader

| 参数 | 约定 |
| --- | --- |
| model_name | 从 `ComfyUI/models/bluepixel/` 注册目录选择完整模型；支持 extra_model_paths.yaml |
| precision | 基线为 BF16 DiT/Qwen3、FP32 DCAE；其他精度必须独立验证 |
| reload_token | 显式刷新同路径更新的模型 |
| 输出 | `BLUEPIXEL_PIPE`，包含受管组件、tokenizer、只读配置和模型指纹 |

加载前检查配置、分片索引、缺失文件和关键 shape。采用本地加载，不在节点导入或执行时自动联网下载。缺失或不匹配的核心权重应报错，不能静默跳过后继续生成。

### Generate

| 参数 | 约定 |
| --- | --- |
| pipe | `BLUEPIXEL_PIPE` |
| image | 可选；连接时图像编辑，未连接时文生图 |
| prompt | 编辑指令或生成描述 |
| seed / steps / guidance | 显式 seed；默认 4 步、CFG 3.0 |
| resolution_mode | match_input / target_pixels / custom |
| target_pixels | 1K、4K 预设及自定义像素预算；实际尺寸写入元数据 |
| width / height | custom 模式有效；校验并明确尺寸取整规则 |
| mu_mode / shift | 默认 empirical / 0；training 要求 shift=0 |
| vae_tiling | 默认开启；提供关闭选项用于基线测试 |
| 输出 | CPU float32 BHWC、范围 [0,1] 的标准 IMAGE；另输出 JSON 格式 STRING 元数据 |

IMAGE batch 表示独立任务，首版逐张处理；不隐式作为多参考图。首版不承诺通用 MODEL/CLIP/VAE/LATENT、KSampler、任意 LoRA、FP8、GGUF 或第三方量化插件兼容。保留 CLI 作为对照入口，新增节点不修改其批处理锁与断点续跑逻辑。

## 4. 分步实现与验证

### P0：冻结环境和原版基线（预计 1–2 天）

**实施内容**

1. 选定 ComfyUI release/commit、Python、PyTorch、CUDA、Diffusers、Transformers 版本，记录依赖锁定清单。
2. 选定正式 BluePixel 权重 revision，记录配置、分片清单和文件指纹。
3. 在独立环境加载真实权重，解决导出版本与 requirements 的潜在兼容问题，不直接覆盖现有 ComfyUI 依赖。
4. 使用现有三个编辑样本运行原 CLI，捕获预处理、prompt embedding、latent、sigma、预测与最终图像。
5. 检查 64/128 尺寸规则和继承的参考图 ID/打包行为，确定兼容基线。

**验证方法与通过标准**

- 所有关键权重完整加载；missing/unexpected keys 必须解释，核心权重不允许缺失。
- 三个编辑样本完成 1K smoke test，大显存设备至少完成一个 4K 样本。
- 基线报告包含实际输出尺寸、组件 dtype、CFG forward 次数、sigma 序列、耗时、GPU 与 CPU 峰值。
- 原版重复运行，测量非确定性范围，为后续浮点门槛提供依据。
- 固定 ComfyUI commit 并记录所用 API 签名，不能把浮动 master 当成发布依赖。

**当前验证情况**：已核查推理代码、依赖声明及本地模型配置；环境未冻结，真实权重 smoke test 和基线采集均待验证。

**交付/关卡**：环境清单、基线报告、黄金中间结果。P0 未通过前，不宣称数值兼容。

### P1：拆分独立推理核心（预计 2–3 天）

**实施内容**

1. 提取 prompt、预处理、latent、scheduler、CFG 和 decode 模块。
2. 保留 chat template、padding、attention mask、hidden-state 层索引。
3. 保留 DCAE scaling、目标与参考 token 拼接、预测裁剪、无条件分支的参考图条件。
4. 每次执行创建独立 scheduler、RNG、callback 与运行状态；pipe 不保存上次任务的中间 tensor。
5. 暂时保留原 CLI 为独立对照，不用同一套新代码同时充当参考与被测实现。

**验证方法与通过标准**

- CPU 单测覆盖 pack/unpack 往返、位置 ID、尺寸边界、非法 steps/shift/mu 组合和 CFG 分支。
- guidance <= 1 每步一次预测，guidance > 1 每步两次预测；保留原实现边界语义。
- 直接注入相同初始 noise，对比每个阶段；仅相同 seed 不足以保证 CPU/GPU 噪声相同。
- token/ID/shape 必须完全一致；FP32 sigma 初始采用 atol=1e-6。
- 浮点结果记录 max error、relative L2 和 cosine；BF16 与图像指标门槛在 P0 后冻结，不能根据新实现误差事后放宽。
- 新 runtime 在固定配置下完成三个样本的逐阶段对齐。

**当前验证情况**：待实现；未运行单测或数值对齐。已静态定位需要提取的函数。

**交付/关卡**：独立 runtime、CPU 单测、逐阶段对齐报告。未通过时先定位首个偏差阶段，不直接比较最终 JPEG。

### P2：组件托管与原生 lowvram（预计 4–6 天）

**实施内容**

1. Qwen3、DiT、DCAE 分别创建 ModelPatcher，组件初始化于 CPU 或受控加载设备。
2. 经 ComfyUI 设备与显存 API，在文本编码、参考图编码、去噪、解码阶段分别请求加载；显存充足时允许管理器保留模型。
3. 清理整条 pipe.to(cuda) 与直接设备迁移路径，不叠加 Accelerate offload hooks。
4. 枚举参数模块与 buffer，以局部工厂或受控转换适配 ComfyUI ops；不使用全局 monkey patch。
5. 审计 Linear/Conv/Embedding/Norm、直接 weight 访问、共享参数、RoPE buffer 与 BatchNorm statistics，保证部分卸载时设备和 dtype 正确。
6. 适配 DCAE encode/decode tiling，基于实际 token 数、精度、CFG 和 tile 配置估算工作显存，并通过实测校准。
7. 在执行边界和长计算阶段检查中断；finally 释放运行 tensor，正确传播中断异常。

**验证方法与通过标准**

- 小模型转换前后 forward 对齐；参数数量、shape 与共享关系符合预期。
- normal 与 lowvram 下均运行真实权重，日志与权重驻留观测证明部分加载确实发生。
- 文本编码器、DiT、DCAE 分别测试，不能只证明 DiT 可卸载。
- 人为制造显存压力后仍能部分加载；不出现 CPU/CUDA tensor 错配。
- 比较完整加载与低显存执行的结果，误差满足冻结门槛。
- tiled 与 non-tiled 比较图像质量，检查接缝、亮度和颜色偏移。
- 切换其他 ComfyUI 模型后可回收 BluePixel 权重，再切回时正常运行。
- 卸载后区分活跃 tensor、PyTorch allocator reserved 与进程显存，不把缓存误判为泄漏。

**当前验证情况**：已阅读上游 ModelPatcher、model_management、ops 相关实现；所有适配与 GPU 验证待执行。仅整组件卸载不能通过本阶段验收。

**交付/关卡**：组件适配层、部分加载证据、显存曲线和质量对比。未通过不得标注“原生 lowvram 支持”。

### P3：节点、缓存与工作流（预计 1–2 天）

**实施内容**

1. 根据 P0 固定版本实现节点注册、输入校验、路径注册和模型缓存。
2. Loader 输出轻量受管 pipe；Generate 的 scheduler、RNG、参数与中间状态每次独立。
3. 模型缓存键包含规范路径、配置/权重指纹、精度和 reload_token；输出缓存由 ComfyUI 管理。
4. 输出标准 IMAGE 和运行元数据，支持 Preview/Save Image。
5. 接入阶段与去噪进度、取消操作，提供编辑与文生图 workflow。
6. 首版 batch 顺序执行并明确 seed 派生规则；多次调用不能污染共享对象。

**验证方法与通过标准**

- 干净 ComfyUI 环境能导入插件，不在 import 时加载权重或初始化大模型。
- UI workflow 和 API prompt 均能完成编辑、文生图、batch 测试。
- 相同输入命中缓存；修改 seed/prompt/resolution 重新执行。
- 权重同路径替换并刷新后不会复用旧模型。
- 文件缺失、shape 不匹配、非法参数有可定位错误。
- 输出设备、dtype、shape、数值范围符合 IMAGE 协议，可预览和保存。
- 取消后进度终止，不输出伪成功结果，下一任务可执行。

**当前验证情况**：待实现；节点注册、UI/API 工作流和缓存行为均未验证。

**交付/关卡**：两个节点、两份 workflow、安装说明和集成测试记录。

### P4：稳定性、性能与发布（预计 2–3 天）

**实施内容**

1. 执行下述 GPU 矩阵与异常恢复用例。
2. 记录冷启动、热启动、各阶段耗时、峰值显存与 CPU RSS。
3. 根据实测确定支持硬件、分辨率和依赖版本，不预先承诺最低显存。
4. 发布模型目录说明、工作流、常见问题、兼容范围和测试报告。

**验证方法与通过标准**

- 连续至少 20 个任务，无持续增长的活跃 GPU tensor 或无界 CPU 缓存。
- 覆盖 BluePixel → 其他模型 → BluePixel、模型目录切换、prompt 和分辨率切换。
- 在文本编码、去噪、解码阶段取消，随后任务成功。
- 人为触发 OOM 后以较小分辨率成功执行；不静默修改用户分辨率重新生成。
- 正常流程和异常流程均不残留可影响后续执行的 scheduler/设备状态。
- 输出实测性能数据；性能预算在 P0 基线后冻结，低显存模式与全驻留模式分别报告。
- 文档中所有“支持”声明均关联实际通过的用例和环境。

**当前验证情况**：待验证；未执行性能、长稳、OOM 或取消恢复测试，暂无可声明的最低显存。

**交付/关卡**：发布包、测试报告、支持矩阵。所有阻断项解决后才发布首版。

## 5. 测试矩阵

| 环境 | 用例 | 预期 | 当前状态 |
| --- | --- | --- | --- |
| CPU CI | 参数、尺寸、ID、scheduler、小模型、节点注册 | 自动测试通过 | 待验证 |
| 可完整加载参考实现的大显存 CUDA GPU | 1K/4K，固定 noise，完整加载 | 数值基线与对齐通过 | 待验证 |
| 24GB CUDA GPU | normal/lowvram，1K/4K tiled | 测定可支持组合与峰值 | 待验证，不保证所有组合成功 |
| 16GB CUDA GPU | 1K lowvram；探索 4K | 明确可运行范围 | 待验证，4K 非首版保证 |
| 固定发布 ComfyUI commit | 安装、队列、缓存、取消、模型切换 | 完整集成通过 | 待验证 |
| 额外 ComfyUI 版本 | 兼容性 smoke test | 通过后加入支持清单 | 待验证 |

数据覆盖：现有三个编辑样本；新增中文、空 prompt、超长 prompt、横图、竖图、极端长宽比、64/128 边界尺寸、无参考图、batch=2、多个 seed。默认 4 步为必测，额外步数和 guidance=1/3 覆盖分支。empirical、training、fixed shift 均覆盖，非法 training + 非零 shift 必须报错。

## 6. 建议目录与测试记录

```text
comfyui_bluepixel/
  __init__.py
  nodes.py
  core/
    pipeline.py
    prompt.py
    image_processing.py
    latents.py
    scheduler.py
  runtime/
    loader.py
    managed_models.py
    comfy_ops_adapter.py
    memory.py
    compatibility.py
  workflows/
    image_edit.json
    text_to_image.json
  tests/
    unit/
    integration/
    gpu/
  pyproject.toml
  README.md
```

测试结果应进入独立报告。大体积权重与 tensor artifact 不直接提交 Git，报告记录 artifact 路径和校验值。

每次验证填写：

| 字段 | 内容 |
| --- | --- |
| 测试 ID / 对应阶段 | 例如 P2-lowvram-text-001 |
| 状态 / 执行日期 | 通过、失败或阻塞；实际日期 |
| 代码 / 模型版本 | 插件 commit、ComfyUI commit、模型 revision 与指纹 |
| 环境 | OS、Python、Torch、CUDA、GPU、显存、CPU RAM、关键依赖 |
| 输入与命令 | 样本、prompt、noise/seed、尺寸、dtype、steps、CFG、shift、tiling、启动参数 |
| 预期与实测 | 指标、阈值、误差、耗时、显存、输出尺寸 |
| 证据 | 日志、报告、图像和 tensor artifact 路径 |
| 失败处理 | 首个偏差阶段、根因、修复 commit、复测结果 |

## 7. 发布检查表

- [ ] P0：环境、模型版本固定，原版真实权重基线通过。
- [ ] P1：核心逻辑与中间 tensor 对齐通过。
- [ ] P2：三个组件托管与真实部分加载通过。
- [ ] P2：DCAE tiling 质量及显存验证通过。
- [ ] P3：两个节点与两个示例 workflow 通过。
- [ ] P3：缓存、刷新、batch、标准 IMAGE 输出通过。
- [ ] P4：连续任务、模型切换、取消、OOM 恢复通过。
- [ ] P4：发布依赖范围、实测显存表、性能报告和已知限制。

总工期估算为 10–16 个工程日，前提是一名熟悉 PyTorch/ComfyUI 的工程师和可用 CUDA 测试环境。主要风险是权重导出版本兼容性，以及 Qwen3/DCAE 的逐层卸载适配；P0/P2 失败时应重新评估工期。

## 8. 参考依据

本仓库：`vins_infer.py`、`requirements.txt`、`README_CN.md`、`assets/test_examples.jsonl`，以及本地模型配置。上游源码在规划时已读取，以下链接随分支变化；实现时必须保存具体 commit 链接。

- [ComfyUI ModelPatcher](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_patcher.py)
- [ComfyUI model_management](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_management.py)
- [ComfyUI ops](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ops.py)
- [ComfyUI Flux 节点与 scheduler](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_flux.py)
- [Diffusers 0.36 Flux2 pipeline](https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/pipelines/flux2/pipeline_flux2.py)

## 9. 本文档自身的验证记录

- 已将实现步骤逐项关联验证方法、通过标准、当前状态和交付关卡。
- 已区分静态核查与真实运行验证，未将任何未执行的 GPU 测试标记为通过。
- 本次仅新增计划文档，不修改推理代码、现有模型文件或依赖。
