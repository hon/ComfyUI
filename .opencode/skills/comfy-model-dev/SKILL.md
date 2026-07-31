---
name: comfy-model-dev
description: ComfyUI 模型实现与推理代码的强制规则。涉及模型 forward、模型加载、state-dict、dtype/设备/内存/VRAM、量化、注意力后端选择、autograd 与模型冻结时必须先加载本 skill。触发词：模型实现、模型代码、model forward、模型加载、state-dict、dtype、VRAM、offload、量化、注意力、autograd、inference_mode。
---

# ComfyUI 模型开发规则

修改 ComfyUI 核心的模型实现、加载或推理代码时，必须遵守以下规则。

## Autograd 与模型冻结

- 不要在 ComfyUI 代码中添加 `torch.no_grad`、`torch.inference_mode` 或推理模式
  辅助包装器。唯一允许的推理模式相关用途是：当训练路径需要梯度时，禁用全局设置的
  推理模式。
- 不要给模型类添加冻结、解冻或可训练性开关。ComfyUI 模型推理时始终视为冻结，
  因此显式冻结功能是多余的，不应添加。
- 从推理模型代码中移除仅训练行为（如 dropout），但保持 checkpoint 与 state-dict
  兼容性。如果删除模块会改变 state-dict 键、模块顺序或 checkpoint 加载行为，
  用无操作（如 `nn.Identity`）替换，而不是直接移除该槽。

## 模型、设备与内存行为

- 把 dtype、设备放置、VRAM 占用和卸载行为当作核心正确性关注点。改动共享执行或
  加载代码时，检查 CPU、CUDA、ROCm、MPS、DirectML、XPU、NPU 和低 VRAM 影响。
- 优先使用 ComfyUI 原生格式和既有量化/卸载辅助函数，而不是添加平行代码路径。
  在 `comfy.quant_ops`、`comfy.model_management`、`comfy.memory_management`、
  `comfy.pinned_memory`、`comfy_aimdo` 和 `comfy-kitchen` 辅助函数已能解决问题时
  使用它们。
- 只要既有优化操作能在不改变预期 dtype、设备、内存或接口行为的情况下支持所需数学
  与张量布局，模型实现就必须使用 Comfy Kitchen 或 ComfyUI 既有操作。这是默认实现
  要求，不是可选后续优化。
- 实现模型数学前，先检查 Comfy Kitchen、`comfy.quant_ops` 和既有 ComfyUI 模型辅助
  函数已暴露的操作。在写本地实现或组合底层 torch 操作前，检查是否有优化的单变体、
  成对变体、融合变体、布局专属变体和量化变体。
- 先使用兼容的优化操作并让模型输入适配其文档化布局，同时保持模型的精确数学。
  若多个优化变体适用，用代表性模型形状做基准并选择最快的合法路径。
- 仅当没有既有优化操作支持所需数学、布局、dtype、设备、autograd 或 patch 契约时
  才添加或保留本地实现。当优化推理操作不提供可微分或可 patch 兼容契约时，保留
  可微分或可 patch 的 fallback。
- 对传给优化操作的参数，使用既有 ComfyUI 转换、卸载和清理辅助函数。保留模型专属
  的 epsilon、缩放、布局、dtype、设备与输出形状行为。
- 优先使用 ComfyUI 共享的优化内核和后端分发器，而不是手写同一操作。删除重复的
  本地内核，并让输入适配共享操作的文档化布局，同时保留模型原始数学与输出契约。
- 所有模型都应使用 ComfyUI 选定的优化注意力函数。将优化后端函数、分发辅助函数和
  按能力选择的可调用对象视为不透明。更上层代码不得通过检查函数身份、名称、模块
  或实现细节来决定行为。
- 对注意力之外的类似模式应用同样的不透明规则：调用方应依赖文档化接口和结果契约，
  而不是依赖下面选择的是哪个后端实现。
- 不要使用仅仅复制既有操作并上转为 float32 的自定义推理操作（如自定义 RMSNorm
  变体）。改用通用 ComfyUI 操作和/或原生 torch 操作。
- 若模型类的 `__init__` 有 `operations` 参数，假定 `operations` 永不为 `None`。
  不要为缺失的 `operations` 对象添加 fallback 分支或默认 torch 操作。
- 不要给模型、模型块或模型操作相关类添加不必要的参数。构造函数和 forward 签名
  只应携带该对象推理时真正需要的值。
- 适当时复用既有模型类、块、操作和辅助模块。实现新版本模型组件前，先搜索既有
  模型代码中是否已有提供该行为的类或辅助函数。
- 检查线性权重形状的模型检测代码只应使用第一维。对于 NVFP4 或其他 4-bit 量化模型，
  第二维可能只有原始大小的一半。
- 模型检测签名必须为其解引用的每个 state-dict 键做保护。不要部分匹配某格式后，
  在提取其配置时抛出意外的 `KeyError`。
- 模型检测检查按从既有或更具体签名到更新或更宽签名的顺序排列。当更高优先级会
  抢走另一模型家族时，把宽泛的新检测器放在通用 fallback 附近。
- 避免在核心推理代码中使用 `einops`。改用原生 torch 张量操作，如 `reshape`、
  `view`、`permute`、`transpose`、`flatten`、`unflatten`、`unsqueeze`、`squeeze`。
- 不要把张量当作通用 Python 数据结构。除非数据必须直接参与张量计算，否则把元数据、
  簿记、计数器、标志、形状数学、padding 数学、索引规划、内存估算和控制流决策保持在
  普通 Python 值中。不要为只用于 Python 侧控制流的结构化元数据创建张量。序列长度、
  累积偏移、切分索引、窗口计数、切片边界和重复计数应从计算时刻起保持为 Python
  int/list。不要把它们构造成 CPU/GPU 张量后再转换回 Python 用于 `split`、
  `tensor_split`、索引规划、循环或缓存键。避免仅为使用张量方法做标量或结构化计算
  而创建临时张量。
- 避免不必要的转换与搬移。保留预期计算 dtype、存储 dtype、bias dtype 和原始张量
  形状元数据。
- 不要把优化后端操作的结果转回其输入 dtype，除非该后端文档化结果契约要求归一化。
  特别地，信任所选优化注意力实现遵守其 dtype 契约。
- 把模型原生 latent 布局处理留在模型或 latent 格式属主内，而不是辅助节点中。
  不要在节点或其他调用方侧适配器中为了满足模型 forward 而折叠、扩展、打包或解包
  latent 维度；模型路径应消费并返回该模型家族的原生 latent 形状。
- DiT 模型应接受不是精确 patch 尺寸倍数的 latent 维度。对每个 patchify 目标或
  参考输入使用 `comfy.ldm.common_dit.pad_to_patch_size`，然后只把目标输出裁剪回
  原始尺寸。
- 避免仅仅替代紧下方张量操作清晰报错的防御性形状和配置检查。仅在检查能在真实边界
  提供实质性更好的上下文或防止静默错误输出时，才添加显式校验。
- 默认假定主模型 forward 的输入已在计算 dtype 中，整数输入（如某些模型 timestep
  张量）除外。不要在模型代码中添加防御性或便利性转换；无效 dtype 管道清晰报错
  好过被不必要的转换掩盖。
- 不属于 op、可能以与计算 dtype 不同 dtype 初始化的裸模型参数，应在 forward 或
  推理代码使用时用 `comfy.ops.cast_to_input` 或 `comfy.model_management.cast_to`
  转换，避免 dtype 不匹配。
- 模型代码不应关心它初始化时是什么 dtype，模型 `__init__` 方法不应包含针对特定
  dtype 的 workaround。dtype workaround 代码（如让模型支持 fp16 计算）属于拥有
  计算策略的执行或模型管理层。
- 模型代码不应做不必要的设备到 CPU 或 CPU 到设备搬移。新分配必须创建在正确的
  设备和 dtype 上；绝不在 CPU 上分配再搬到 GPU，也不要在一种 dtype 分配再转成
  另一种。
- 模型代码本身不应做内存管理。加载、卸载、offload、设备移动、VRAM 策略、缓存
  生命周期和清理属于相关模型管理和执行层，而不是模型实现内部。
- 不要添加跨执行持续存在的全局、模块级、类级、单例或模型拥有的张量或其他大内存
  存储。临时缓存必须限定在单次执行或 forward/encode/decode 调用内：在属主的顶层
  调用中分配，显式沿调用栈传递，调用返回时丢弃。
- 临时缓存遵循 Wan VAE 时间缓存模式：为 encode/decode 操作创建本地缓存
  （如 `feat_map`），传入需要它的块，不要保留在模型上或全局状态中。
- 模型初始化代码中，为从模型 state dict 填充的参数/缓冲占位优先用 `torch.empty`，
  而不是用 `torch.zeros` 等零初始化。如果某分配不来自 state dict 且对推理无用，
  就不要包含它。
- 存储在模型 state dict 中并由其填充的 `nn.Parameter` 张量应用 `torch.empty`
  初始化，而不是零、随机或其他有意义初始化。
- 模型初始化应描述模块结构，而不是伪造 checkpoint 拥有的张量内容。从 state dict
  加载的参数和缓冲不得手动初始化、重新赋值或用 fallback 值填充，除非在无 checkpoint
  键时该值确实被使用。
- 切分大张量时，若切片生命周期超过当前函数作用域，复制该切片。当小副本能更早
  释放内存时，不要持有指向大后备张量的长生命周期视图。
- 数学上自然匹配时使用融合或复合 torch 操作（如 `addcmul`）。在不遮蔽代码或改变
  dtype/设备行为的前提下，减少 Python 和 torch 分发开销是合法优化。
- 尽量避免跨不同执行持续存在的缓存。仅当持久缓存占用极少内存且有清晰属主和失效
  逻辑时才可接受。
- 优化时偏好小而可衡量的改动：更少分配、更少设备搬移、更低峰值内存、更好批处理，
  或使用更快的既有后端操作。
