# 验证记录

按 `COMFYUI_INTEGRATION_PLAN.md` 的记录格式，登记本仓库已实际执行的验证。
环境统一为：Linux, Python 3.10, torch 2.9.0+cu129, diffusers 0.38.0（本地环境）,
transformers 4.57.1, GPU 96GB（CUDA sm86, 单卡）。
模型：`/data/vjuicefs_ai_camera_lgroup_ql/public_data/72225263/hf_upload/BluePixel-Turbo`
（fingerprint `7d3ccd3cd73e3cea`，transformer 9.08B 参数 bf16，Qwen3-8B 文本编码器，DCAE fp32，scaling_factor 0.4538）。

## 单元测试（CPU）

| 字段 | 内容 |
| --- | --- |
| 测试 ID | U1-core-unit |
| 状态 / 日期 | 通过 / 2026-11-10 |
| 代码版本 | 本仓库工作区（未打 tag） |
| 命令 | `python -m pytest bluepixel_comfyui/tests/unit -q` |
| 预期与实测 | 43 passed, 0 failed；覆盖 pack/unpack 往返、latent/image id 布局、sigma 序列 vs diffusers 调度器（atol 1e-5，实测 7.8e-8）、尺寸规则、mu 校验、CFG 分支（guidance=1 单次 forward / 3.0 双次）、种子派生、确定性、目录校验与指纹 |

## GPU 冒烟（真实权重）

| 字段 | 内容 |
| --- | --- |
| 测试 ID | G1-real-smoke |
| 状态 / 日期 | 通过 / 2026-11-10 |
| 命令 | `python bluepixel_comfyui/tests/gpu/test_real_weights.py --model-dir <MODEL>` |
| 预期与实测 | 目录校验通过；prompt embeds (1,512,12288)；sigma 序列与 diffusers max_err=0.0；768x1280 编辑 2.9s；同 seed 两次输出 bit-equal；峰值显存 43.2 GB（全组件驻留 bf16+fp32） |

## 参考实现逐阶段对齐（真实权重）

| 字段 | 内容 |
| --- | --- |
| 测试 ID | G2-reference-parity |
| 状态 / 日期 | 通过 / 2026-11-10 |
| 命令 | `python bluepixel_comfyui/tests/gpu/test_reference_parity.py --model-dir <MODEL>` |
| 输入 | test_input.jpg（3600x5998），prompt 灰毛衣改酒红，seed 42，steps 4，CFG 3.0，target 1<<20 |
| 预期与实测 | 8/8 transformer forward（4 步 × pos/neg）rel_l2 = 0.000000（bit-exact）；最终图像 PSNR 59.06 dB，max_abs 0.0020（uint8 量化路径差异） |

对齐过程中定位并修复的偏差（均以 bit-exact 收敛）：

1. 参考图 PIL 往返：torchvision `to_pil_image` 对 float 截断而非四舍五入，改为显式 round→uint8。
2. Euler 更新数值：diffusers `scheduler.step` 先把 sample upcast fp32、`dt`（fp32 0-dim）与 bf16 输出相乘得 bf16 积再 fp32 累加；复刻该混合精度路径。
3. sigma 序列：diffusers 在 numpy fp32 数组上做 exponential shift（fp64 标量），改为 numpy 路径，序列 bit-exact。
4. timestep 约定：参考实现是 `t.to(bf16)/1000`（先 bf16 再除），不是 `bf16(t/1000)`；在 768x1280 下二者相差 1 bf16 ulp，已按参考路径复刻。

## ComfyUI 端到端（真实服务）

| 字段 | 内容 |
| --- | --- |
| 测试 ID | G3-comfyui-e2e |
| 状态 / 日期 | 通过 / 2026-11-10 |
| 环境 | ComfyUI 0.44.0（commit f4f3ded）部署于 /root/comfyui-many/ComfyUI，端口 4344，--disable-cuda-malloc（GPU 驱动不满足 cudaMallocAsync 要求） |
| 模型注册 | models/bluepixel/BluePixel-Turbo（符号链接）；custom_nodes/bluepixel_comfyui（符号链接） |
| 命令 | `python main.py --listen 0.0.0.0 --port 4344`；API /prompt 提交 Loader→LoadImage→Generate→SaveImage |
| 预期与实测 | 节点导入 0.0s（不加载权重）；/object_info 注册两节点并枚举模型；编辑工作流 113.9s（含模型加载，输出 3200x5312 PNG）；同配置第二次执行命中节点缓存 3.4s；1K 档输出 768x1280 与独立运行一致 |

修复记录：HF 模型（transformers/diffusers ModelMixin）的 `device` 为只读 property，
与 comfy.ModelPatcher 的 `model.device = ...` 写入冲突；新增 DeviceAttrShim 包装
（`runtime/managed_models.py`）后通过。

## 纯编辑模型改版（G5）

| 字段 | 内容 |
| --- | --- |
| 测试 ID | G5-edit-only |
| 状态 / 日期 | 通过 / 2026-11-10 |
| 变更 | BluePixel 是纯图像编辑模型，`image` 输入改为**必选**（从 optional 移入 required）；缺图时 ComfyUI 校验层直接拒绝入队（`required_input_missing: image`）；节点内保留二次防御性检查；删除 t2i 示例 workflow |
| 预期与实测 | /object_info 显示 required 含 image、optional 为空；无 image 的 prompt 返回 HTTP 400 node_errors.3.required_input_missing；正常编辑工作流 121s 通过（输出 3200×5312）；43 个 CPU 单测通过 |

## 固定 4K 输出尺寸改版（G4）

| 字段 | 内容 |
| --- | --- |
| 测试 ID | G4-fixed-4k |
| 状态 / 日期 | 通过 / 2026-11-10 |
| 变更 | BluePixelGenerate 节点移除 resolution_mode / target_pixels / custom_pixels / width / height 参数；输出锁定 4K 像素预算（4096×4096 = 16777216 px），编辑保持输入比例 64 网格选尺寸，小图 LANCZOS 预放大到预算尺寸（`run_bluepixel` 新增 pre-resize 步骤），t2i 输出 4096×4096 |
| 命令 | 重启 ComfyUI 后经 /prompt 执行两个 example workflow |
| 预期与实测 | edit：test_input.jpg（3600×5998）→ 3200×5312（17.0M px），131s 含加载；t2i：4096×4096（16.78M px），15s；43 个 CPU 单测全部通过 |

## 模型目录注册改版（G6）

| 字段 | 内容 |
| --- | --- |
| 测试 ID | G6-folder-registration |
| 状态 / 日期 | 通过 / 2026-11-10 |
| 变更 | Loader 节点的 model_name 改为走 ComfyUI 标准 folder_paths 机制：注册 `bluepixel` 类别目录（扩展名过滤 `{'.json'}`，以 `<模型名>/model_index.json` 为模型标记），下拉选项来自 `folder_paths.get_filename_list`（自动合并 extra_model_paths.yaml 注册目录、带框架缓存）；`_resolve_model_dir` 收紧为仅接受注册列表内的名字，绝对路径与 `..` 穿越一律拒绝 |
| 预期与实测 | /object_info 下拉返回 ['BluePixel-Turbo']；绝对路径/相对穿越 3 组用例全部 FileNotFoundError；编辑工作流 141s 成功（输出 3200×5312）；43 个 CPU 单测通过 |

## 新版推理内核（vins_infer fb1366a）+ INT8 支持（G7）

| 字段 | 内容 |
| --- | --- |
| 测试 ID | G7-vinfer-fb1366a |
| 状态 / 日期 | 通过 / 2026-11-12 |
| 变更 | 推理核心同步到 vins_infer fb1366a：①去掉 CFG（guidance=None 单次前向）②bucket 分辨率系统（16 比例、4096²面积、64 网格、按对数列距选择，参考图 BICUBIC resize+中心裁剪）③显式 denoising_step_list `[1000,750,500,250]` ④CPS 调度（eta 0.9、fp32 x0）⑤training mu 上限 4096 ⑥INT8 weight-only transformer（Int8Linear，153 层 per-channel scale）自动识别 ⑦base_model 分离（INT8 的 vae/text/tokenizer/scheduler 取 FP32 目录）⑧全局 RNG 种子（CPS 噪声可复现）⑨VAE tiling 默认关闭 |
| 命令 | `python -m pytest bluepixel_comfyui/tests/unit -q`；ComfyUI /prompt 端到端 |
| 单测 | 53 passed（新增：CPS 步进/eta 随机性/step_index、bucket 面积与对数选择（与参考逐项一致）、preprocess 与参考逐位一致、Int8Linear 前向数学、id 生成与参考一致、engine 无 CFG 前向次数/bucket 输出尺寸/确定性/元数据） |
| GPU 实测 FP32 | 3072×5440，50s（含加载），峰值约 47GB |
| GPU 实测 INT8 | 153 Int8Linear 层，3072×5440，30s（含加载），峰值约 20GB |
| 数值对齐 | **单进程 stage diff**：vins_infer BluePixelPipeline 与插件 engine 共用同一组权重，4 步 v_pred 全部 `rel_l2 = 0.000e+00`，最终图像 PSNR 138 dB（逐位一致） |
| 跨进程差异说明 | 插件经 ComfyUI `LoadImage` 取图（pyav 解码 JPEG），参考 CLI 用 PIL 解码：同一 JPEG 的像素差异 max 24 / mean 0.58，导致端到端 mean≈1.19。已单独验证：把 pyav 解码后的像素喂给插件离线重建，与线上 ComfyUI 输出**逐位一致**（PSNR 138 dB），证明算法移植无偏差 |

## 显存修复：分阶段加载 + VAE 切块（G8）

| 字段 | 内容 |
| --- | --- |
| 测试 ID | G8-lowvram-staged |
| 状态 / 日期 | 通过 / 2026-11-17 |
| 背景 | 用户实测 INT8 4K 峰值 ~40GB，与本地脚本 ~20GB 不符，怀疑加载器把 int8 转成 bf16/fp32 |
| 排查结论 | 加载器无问题：transformer buffers 为 `torch.int8` 9.08GB + fp32 scale 0.01GB，`Int8Linear.weight.dtype == torch.int8`。峰值高是三个组件同时常驻 + 4K VAE 编码未切块 |
| 实测分解 | Qwen3 16.4GB / DCAE 2.7GB / INT8 DiT 9.1GB；4K VAE encode 不切块 +21.6GB、切块 +0.34GB；DiT forward +3.5GB；文本编码 +1.6GB |
| 参考 CLI 同配置 | 4K INT8 不切块峰值 49.8GB（与节点修复前一致，说明当时无额外泄漏） |
| 修复 | ①`core/pipeline.py` 增加 `stage_cb`（text_encode/image_encode/denoise/decode）②`nodes.py` 按阶段加载并在切换时卸载，结束释放 ③`unload_patcher` 改为通过 ComfyUI `current_loaded_models` 注册表驱逐（`is_clone` 匹配），否则注册表残留导致权重不释放 ④`vae_tiling` 默认改回 True |
| 预期与实测 | 4K INT8 峰值 **49.8 → 20.3GB**；运行后残留 **29.4 → 1.1GB**（释放 19.2GB）；耗时 30s → 63s（多 3 次权重搬运）；53 个单测通过 |
| 发现的环境问题 | 用户此前通过 Manager 安装的是 `custom_nodes/bluepixel-comfyui/`（Registry 0.1.0 独立副本），本地修改不生效；已删除副本改回软链 |

## 说明与边界

- 参考图人脸 artifact（768x1280 / 4 步 / seed 42）：参考 pipeline 输出同图（G2 保存 `ref.jpg`/`ours.jpg` 对照），属模型在该分辨率/步数下的自身行为，非移植缺陷。
- lowvram 逐层部分加载（计划 P2 验收项）：ops 适配层已实现（`runtime/comfy_ops_adapter.py`）并接入 ModelPatcher，但未在低显存 GPU 上做部分加载实测，暂不声明"原生 lowvram 支持"。
- ComfyUI 端到端已验证：节点注册、模型下拉、FP32/INT8 加载与推理、输出保存；UI 交互（缓存/取消/切换）部分验证。
- 跨进程像素差异来自 LoadImage 的 pyav 解码，非实现偏差；如需与 CLI 完全一致的像素，需自行预转 PNG 输入。
- bucket 面积固定 4096²，不支持自定义分辨率。
