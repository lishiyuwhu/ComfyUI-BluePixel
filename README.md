# ComfyUI BluePixel

[BluePixel-Turbo](https://huggingface.co/openvivo/BluePixel-Turbo)（基于 Flux2-Klein + DCAE 的 4K 图像编辑蒸馏模型）的 ComfyUI 节点插件。

第一版提供两个节点，保留 `vins_infer.py` 推理语义，并接入 ComfyUI 的模型托管、进度与取消机制：

```text
BluePixel Pipeline Loader ── BLUEPIXEL_PIPE ──┐
                                              ├─ BluePixel Generate ─ IMAGE ─ Save/Preview Image
        Load Image ────────────── IMAGE ──────┘
```

**BluePixel 是纯图像编辑模型**：`image` 输入为必选，不支持文生图（t2i）。

## 安装

**方式 A：Comfy Registry（推荐）**

```bash
comfy node install bluepixel-comfyui
```

**方式 B：手动**

```bash
cd ComfyUI/custom_nodes
git clone <this repo> ComfyUI-BluePixel
pip install -r ComfyUI-BluePixel/requirements.txt
```

依赖只包含 `diffusers` / `transformers` / `peft` / `pillow` / `numpy` / `safetensors`；
**不含 torch**（由 ComfyUI 环境提供），因此不会改动现有 PyTorch/CUDA 版本。

## 模型

本插件**不包含模型权重**，需单独下载。

### 下载

```bash
# FP32 权重（必需）
hf download openvivo/BluePixel-Turbo --local-dir BluePixel-Turbo

# 可选：INT8 量化 transformer（显存约 20GB，FP32 约 47GB）
# 仅 transformer/ 目录不同，其余组件复用 FP32
```

### 存放目录

```
ComfyUI/models/bluepixel/
├── BluePixel-Turbo/            # FP32（完整 diffusers 布局，必需）
│   ├── transformer/
│   ├── vae/
│   ├── text_encoder/
│   ├── tokenizer/
│   └── scheduler/
└── BluePixel-Turbo-int8/       # INT8（可选，只需 transformer/）
    └── transformer/
```

模型目录必须直接位于 `models/bluepixel/` 下（深度 1），或以同样结构通过
`extra_model_paths.yaml` 注册：

```yaml
bluepixel:
  base_path: /path/to/your/models/
  bluepixel: |
    BluePixel-Turbo
    BluePixel-Turbo-int8
```

权重只从本地读取，**运行时不联网下载**；加载前会校验目录结构、分片索引与关键配置
（`in_channels=128`、`vae.scaling_factor`）。

## 节点

### BluePixel Pipeline Loader

| 参数 | 说明 |
| --- | --- |
| model_name | `ComfyUI/models/bluepixel/` 下的模型目录（FP32 或 INT8，自动识别） |
| base_model_name | INT8 模型时指向 FP32 目录（vae/text_encoder/tokenizer/scheduler 来自这里）；FP32 模型填自身即可 |
| precision | `bf16`（基线）/ `fp16` / `fp32`；作用于 Qwen3 与 latent dtype，DiT/DCAE 固定 fp32 |
| int8_compute_dtype | INT8 transformer 的计算精度：`fp32`（推荐）/ `bf16` |
| vae_tiling | 默认**关闭**（与参考 CLI 一致，保证像素对齐）；显存不足时可开启 |
| reload_token | 非空时绕过缓存强制重载同路径模型 |

**INT8 自动识别**：若 `<model>/transformer/config.json` 含 `quantization_config`，则用
`Int8Linear`（weight-only int8 + per-output-channel scale）构建 transformer，共 153 层；
其余组件从 `base_model_name` 读取。INT8 显存约 20GB（FP32 约 47GB）。

输出 `BLUEPIXEL_PIPE`：ComfyUI ModelPatcher 托管的 DiT/Qwen3/DCAE + tokenizer + 配置指纹。

### BluePixel Generate

| 参数 | 说明 |
| --- | --- |
| image | **必选**，待编辑图像 |
| prompt | 编辑指令 |
| seed | 随机种子（同时作为全局 RNG 种子，保证 CPS 噪声可复现） |
| steps | 去噪步数，默认 4（上限 4，对应 denoising list `[1000,750,500,250]`） |
| cps_eta | CPS 随机性 0–1，默认 0.9（0 = 确定性 Euler） |
| mu_mode | `training`（默认，seq-len 上限 4096）/ `empirical` |

**无 CFG**：蒸馏 checkpoint 固定 `guidance=None` 单次前向，每步只跑一次 transformer。
（旧版插件的 guidance/shift 参数已移除。）

**输出尺寸：bucket 系统（4K 面积预算，节点内固定）**

- 16 个预设宽高比（1:1, 1:4, 1:8, 2:3, 3:2, 3:4, 4:1, 4:3, 4:5, 5:4, 8:1, 9:16, 16:9, 21:9→7:3），
  面积预算 4096×4096 = 16.78M 像素，边长按 64 向下取整
- 按输入图像宽高比的**对数距离**选择最接近的 bucket（与参考 CLI 完全一致）
- **小图自动放大**：参考图用 BICUBIC 等比放大 + 中心裁剪到 bucket 尺寸
- 大图自动缩小
- 实际输出写入 metadata_json：`width`/`height`/`bucket_ratio`/`bucket_id`/`image_seq_len`

输出：标准 `IMAGE`（CPU float32 BHWC，[0,1]）+ `metadata_json`（尺寸、mu、seq_len、seed、模型指纹等）。

Batch IMAGE 逐张顺序处理（每张派生 seed = base_seed + index）。每步执行/长阶段检查 ComfyUI 取消（Cancel）。

## Workflows

`workflows/` 下提供两份可直接加载的工作流（拖入 ComfyUI 画布即可）：

| 文件 | 用途 |
| --- | --- |
| `workflows/image_edit_4k_fp32.json` | FP32 模型编辑（~47GB 显存） |
| `workflows/image_edit_4k_int8.json` | INT8 模型编辑（~20GB 显存） |

两份工作流都是 `BluePixel Pipeline Loader → Load Image → BluePixel Generate → Save Image`，
默认参数：4 步、`cps_eta=0.9`、`mu_mode=training`、4K bucket 面积。

## 数值对齐（已验证）

与参考实现 `bluepixel/vins_infer.py`（FluxKleinPipeline）真实权重逐阶段对齐（详见 `VERIFICATION.md`）：

- prompt embeds / text ids / 参考图 latent / 初始噪声：bit-exact
- sigma 序列：与 diffusers 调度器 bit-exact
- 4 步 4 次 transformer forward（无 CFG）：bit-exact（`v_pred` rel_l2 = 0）
- 同进程最终图像 PSNR 138 dB（逐位一致）
- INT8 与 FP32 两条路径均已验证；插件输出与参考 CLI 的差异仅来自
  ComfyUI `LoadImage` 用 pyav 解码 JPEG（CLI 用 PIL），在相同输入像素下逐位一致

## 发布到 Comfy Registry

仓库已按 Registry 规范组织（根目录 `__init__.py` / `nodes.py`，`pyproject.toml` 含
`[tool.comfy]`，`.comfyignore` 排除测试与开发文档）。

发布前需要：

1. 在 https://registry.comfy.org 创建 publisher，取得 `PublisherId`
2. 把 `pyproject.toml` 里的 `PublisherId = "__PUBLISHER_ID__"` 替换为真实值
3. 每次发布递增 `version`

```bash
pip install comfy-cli
comfy node publish          # 交互式输入 Registry API Key
```

GitHub Actions 自动发布可参考官方文档（`.github/workflows/publish_action.yml`）。

## 测试

```bash
# CPU 单元测试（无需模型权重）
python -m pytest bluepixel_tests/unit -q

# GPU 冒烟（需真实权重）
python bluepixel_tests/gpu/test_real_weights.py \
    --model-dir /path/to/BluePixel-Turbo

# 参考实现逐阶段对齐（需真实权重 + /root/code/bluepixel）
python bluepixel_tests/gpu/test_reference_parity.py \
    --model-dir /path/to/BluePixel-Turbo
```

## 已知边界

- 首版不承诺通用 MODEL/CLIP/VAE/LATENT、KSampler、LoRA、FP8/GGUF 兼容
- FP32 实测约 47GB、INT8 约 20GB 常驻；lowvram 逐层流式加载依赖 ComfyUI ModelPatcher，未在低显存卡上验证
- Qwen3 文本编码器常驻约 16 GB（bf16）；显存不足时 ComfyUI 会自动卸载再加载
- 输出尺寸由 bucket 决定，不支持任意分辨率；需要其他比例请改 bucket 逻辑
- 参考图先按 target_pixels 缩放再走 64 网格搜索（vins 语义）；输出最小 64px、最大 8192px

## 许可

代码遵循上游仓库许可；模型权重遵循 OpenAtom Model License（见上游 BluePixel 仓库）。
