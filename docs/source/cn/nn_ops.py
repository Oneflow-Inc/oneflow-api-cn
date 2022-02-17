import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.nn.Dropout,
    r"""Dropout(p:=0.5, inplace=False, generator=None)
    
    在训练期间，使用来自伯努利分布 (Bernoulli distribution) 的样本，以概率 :attr:`p` 随机将输入张量的一些元素归零。
    在每次 forward call 时，所有的通道都将独立归零。

    如 "Improving neural networks by preventing co-adaptation of feature detectors" 所述，此方法已被证明可以有效用于正则化 (regularization) 和防止神经元协同适应 (co-adaptation of neurons) 。

    此外，在训练期间，输出按因数 :math:`\frac{1}{1-p}` 进行缩放。这意味着在评估过程中只计算一个恒等函数。

    参数：
        - **p** (float): 元素归零的概率。默认：0.5
        - **inplace** (bool): 是否执行 in-place 操作。默认为： ``False`` 

    形状：
        - **Input** : :math:`(*)` 输入可以是任何形状
        - **Output** : :math:`(*)` 输出与输入形状相同

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> m = flow.nn.Dropout(p=0)
        >>> x = flow.tensor([[-0.7797, 0.2264, 0.2458, 0.4163], [0.4299, 0.3626, -0.4892, 0.4141], [-1.4115, 1.2183, -0.5503, 0.6520]],dtype=flow.float32)
        >>> y = m(x)
        >>> y #doctest: +ELLIPSIS
        tensor([[-0.7797,  0.2264,  0.2458,  0.4163],
                [ 0.4299,  0.3626, -0.4892,  0.4141],
                [-1.4115,  1.2183, -0.5503,  0.6520]], dtype=oneflow.float32)
 

    """
)

reset_docstr(
    oneflow.nn.ELU,
    r"""ELU(alpha=1.0, inplace=False)
    
    应用以下逐元素公式：

    .. math::

        \text{ELU}(x) = \begin{cases}
				x & \text{ if } x \gt 0  \\
                \alpha*(exp(x)-1) & \text{ if } x \le 0 \\
    		    \end{cases}

    参数：
        - **alpha** : ELU 公式的 :math:`\alpha` 值。默认：1.0
        - **inplace** : 是否执行 in-place 操作。默认： ``False`` 

    形状：
        - **Input** : :math:`(N, *)` 其中 `*` 表示任意数量的额外维度
        - **Output** : :math:`(N, *)` 形状与输入一致

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([-0.5, 0, 0.5], dtype=flow.float32)
        >>> elu = flow.nn.ELU()

        >>> out = elu(input)
        >>> out
        tensor([-0.3935,  0.0000,  0.5000], dtype=oneflow.float32)

    """
)
    
reset_docstr(
    oneflow.nn.Embedding,
    r"""Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=None, scale_grad_by_freq=False, sparse=False, _weight=None)
    
    一个简单的查找表，用于存储固定字典和大小的嵌入。

    该模块通常用于储存词嵌入并索引它们。该模块的输入是索引列表，输出是相应的词嵌入。

    参数：
        - **num_embeddings** (int): 嵌入字典的大小
        - **embedding_dim** (int): 每个嵌入向量的大小
        - **padding_idx** (int, 可选的): 如果设定了此参数，则 :attr:`padding_idx` 处的元素不会影响梯度；因此，
                                    在训练期间不会更新 :attr:`padding_idx` 的嵌入向量，它仍然是一个固定的 `pad` 。
                                    对于新构建的嵌入， :attr:`padding_idx` 处的嵌入向量将默认为全零，
                                    但可以更新为另一个值以用作填充向量。
    
    示例：

    .. code-block:: python
        
        >>> import oneflow as flow
        
        >>> indices = flow.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=flow.int)
        >>> m = flow.nn.Embedding(10, 3)
        >>> y = m(indices)

    """
)

reset_docstr(
    oneflow.nn.FakeQuantization,
    r"""FakeQuantization(quantization_formula='google', quantization_bit=8, quantization_scheme='symmetric')
    
    在训练时间内模拟量化 (quantize) 和反量化 (dequantize) 操作。

    输出将计算为：

        若 quantization_scheme == "symmetric":

        .. math::

            & quant\_max = 2^{quantization\_to\_bit - 1} - 1

            & quant\_min = -quant\_max

            & clamp(round(x / scale), quant\_min, quant\_max) * scale

        若 quantization_scheme == "affine":

        .. math::

            & quant\_max = 2^{quantization\_to\_bit} - 1

            & quant\_min = 0

            & (clamp(round(x / scale + zero\_point), quant\_min, quant\_max) - zero\_point) * scale

    参数：
        - **quantization_bit** (int): 量化输入为 uintX / intX ， X 可以在范围 [2, 8] 中。默认为 8
        - **quantization_scheme** (str): "symmetric" 或 "affine" ， 量化为有符号/无符号整数。 默认为 "symmetric"
        - **quantization_formula** (str): 支持 "google" 或 "cambricon"

    返回类型：
        oneflow.Tensor: 量化和反量化操作后的输入张量

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input_tensor = flow.rand(2, 3, 4, 5, dtype=flow.float32) - 0.5 
        
        >>> quantization_bit = 8
        >>> quantization_scheme = "symmetric"
        >>> quantization_formula = "google"
        >>> per_layer_quantization = True

        >>> min_max_observer = flow.nn.MinMaxObserver(quantization_formula=quantization_formula, quantization_bit=quantization_bit,
        ... quantization_scheme=quantization_scheme, per_layer_quantization=per_layer_quantization)
        >>> fake_quantization = flow.nn.FakeQuantization(quantization_formula=quantization_formula, quantization_bit=quantization_bit, 
        ... quantization_scheme=quantization_scheme)

        >>> scale, zero_point = min_max_observer(
        ...    input_tensor,
        ... )

        >>> output_tensor = fake_quantization(
        ...    input_tensor,
        ...    scale,
        ...    zero_point,
        ... )

    """
)

reset_docstr(
    oneflow.nn.Flatten,
    r"""Flatten(start_dim=1, end_dim=-1)
    
    将 tensor 指定连续范围的维度展平。用于：nn.Sequential 。

    参数：
        - **start_dim** (int): 展平开始的维度（默认为 1）
        - **end_dim** (int): 展平结束的维度（默认为 -1）
    
    示例：

    .. code-block:: python 

        >>> import oneflow as flow
        >>> input = flow.Tensor(32, 1, 5, 5)
        >>> m = flow.nn.Flatten()
        >>> output = m(input)
        >>> output.shape
        oneflow.Size([32, 25])

    """
)

reset_docstr(
    oneflow.nn.FusedBatchNorm1d,
    r"""FusedBatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    在 2D 或 3D 输入上应用 Fused Batch Normalization ，公式为：
    
    .. math:: 

        out = ReLU(BatchNorm(input) + addend)

    Batch Normalization 的公式为：

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    逐维度小批量计算均值和标准差，其中 :math:`\gamma` 和 :math:`\beta`  是大小为 `C` （ `C` 是输入大小）
    的可学习参数向量。默认情况下， :math:`\gamma` 的元素被设置为 1 ，并且 :math:`\beta` 被设置为 0 。通过有偏估计器 
    (biased estimator) 计算标准差，相当于 `torch.var(input, unbiased=False)` 。

    默认情况下，该层在训练期间不断估计计算的均值和方差，然后在评估期间用于归一化（normalization）。
    运行估计期间， :attr:`momentum` 为 0.1。

    如果将 :attr:`track_running_stats` 设置为 ``False`` ，则该层不会继续进行估计，
    而是在评估期间也使用批处理统计信息。

    .. note::
        参数 :attr:`momentum` 与优化器 (optimizer) 中使用的参数和传统的动量 (momentum) 概念都不同。
        在数学上，这里运行统计的更新规则是 :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t` ，
        其中 :math:`\hat{x}` 是估计的统计量，:math:`x_t` 是新的观察值。
    
    因为 Batch Normalization 是在维度 `C` 上完成的，并在 `(N, L)` 切片上计算统计数据，所以学术上普遍称之为 Temporal Batch Normalization 。

    参数：
        - **num_features** : 来自大小为 :math:`(N, C, L)` 的预期输入的 :math:`C` 或者来自大小为 :math:`(N, L)` 的输入的 :math:`L` 
        - **eps** : 为数值稳定性而添加到分母的值。默认： 1e-5
        - **momentum** : 用于 :attr:`running_mean` 和 :attr:`running_var` 计算的值。可以设置为 ``None`` 以计算累积移动平均。默认： 0.1
        - **affine** (bool, 可选): 如果为 ``True`` ，则该模块具有可学习的仿射参数。默认： ``True``
        - **track_running_stats** (bool, 可选): 如果为 ``True`` ，则此模块跟踪运行均值和方差。如果为 ``False`` ，此模块不跟踪此类统计信息，同时初始化统计缓冲区 :attr:`running_mean` 和 :attr:`running_var` 为 ``None`` 。当这些缓冲区为 ``None`` 时， 此模块在训练和评估模式中始终使用 batch statistics 。默认： ``True`` 

    形状：
        - **Input** : :math:`(N, C)` 或 :math:`(N, C, L)`
        - **Output** : :math:`(N, C)` 或 :math:`(N, C, L)` （与输入形状相同）

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> x = flow.randn(20, 100).to("cuda") # 目前 GPU 支持 FusedBatchNorm 。
        >>> m = flow.nn.FusedBatchNorm1d(num_features=100, eps=1e-5, momentum=0.1).to("cuda")
        >>> y = m(x, addend=None)

    """
)

reset_docstr(
    oneflow.nn.FusedBatchNorm2d,
    r"""FusedBatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    在 4D 输入上应用 Fused Batch Normalization ，公式为：
    
    .. math:: 

        out = ReLU(BatchNorm(input) + addend)

    Batch Normalization 的公式为：

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    逐维度小批量计算均值和标准差，其中 :math:`\gamma` 和 :math:`\beta`  是大小为 `C` （ `C` 是输入大小）
    的可学习参数向量。默认情况下， :math:`\gamma` 的元素被设置为 1 ，并且 :math:`\beta` 被设置为 0 。通过有偏估计器 
    (biased estimator) 计算标准差，相当于 `torch.var(input, unbiased=False)` 。

    默认情况下，该层在训练期间不断估计计算的均值和方差，然后在评估期间用于归一化（normalization）。
    运行估计期间， :attr:`momentum` 为 0.1 。

    如果将 :attr:`track_running_stats` 设置为 ``False`` ，则该层不会继续进行估计，
    而是在评估期间也使用批处理统计信息。
    
    .. note::
        参数 :attr:`momentum` 与优化器 (optimizer) 中使用的参数和传统的动量 (momentum) 概念都不同。
        在数学上，这里运行统计的更新规则是 :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t` ，
        其中 :math:`\hat{x}` 是估计的统计量，:math:`x_t` 是新的观察值。

    因为 Batch Normalization 是在维度 `C` 上完成的，并在 `(N, H, W)` 切片上计算统计数据，所以学术上普遍称之为 Temporal Batch Normalization 。

    参数：
        - **num_features** : 来自大小为 :math:`(N, C, H, W)` 的预期输入的 :math:`C` 
        - **eps** : 为数值稳定性而添加到分母的值。默认： 1e-5
        - **momentum** : 用于 :attr:`running_mean` 和 :attr:`running_var` 计算的值。可以设置为 ``None`` 以计算累积移动平均。默认： 0.1
        - **affine** (bool, 可选): 如果为 ``True`` ，则该模块具有可学习的仿射参数。默认： ``True``
        - **track_running_stats** (bool, 可选): 如果为 ``True`` ，则此模块跟踪运行均值和方差，如果为 ``False`` ，此模块不跟踪此类统计信息，同时初始化统计缓冲区 :attr:`running_mean` 和 :attr:`running_var` 为 ``None`` 。当这些缓冲区为 ``None`` 时， 此模块在训练和评估模式中始终使用 batch statistics 。默认： ``True`` 

    形状：
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` （与输入形状相同）

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> x = flow.randn(4, 2, 8, 3).to("cuda") # 目前 GPU 支持 FusedBatchNorm 。
        >>> m = flow.nn.FusedBatchNorm2d(num_features=2, eps=1e-5, momentum=0.1).to("cuda")
        >>> y = m(x, addend=None)

    """
)

reset_docstr(
    oneflow.nn.FusedBatchNorm3d,
    r"""FusedBatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    在 5D 输入上应用 Fused Batch Normalization ，公式为：
    
    .. math:: 

        out = ReLU(BatchNorm(input) + addend)

    Batch Normalization 的公式为：

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    逐维度小批量计算均值和标准差，其中 :math:`\gamma` 和 :math:`\beta`  是大小为 `C` （ `C` 是输入大小）
    的可学习参数向量。默认情况下， :math:`\gamma` 的元素被设置为 1 ，并且 :math:`\beta` 被设置为 0 。通过有偏估计器 
    (biased estimator) 计算标准差，相当于 `torch.var(input, unbiased=False)` 。

    默认情况下，该层在训练期间不断估计计算的均值和方差，然后在评估期间用于归一化（normalization）。
    运行估计期间， :attr:`momentum` 为 0.1 。

    如果将 :attr:`track_running_stats` 设置为 ``False`` ，则该层不会继续进行估计，
    而是在评估期间也使用批处理统计信息。

    .. note::
        参数 :attr:`momentum` 与优化器 (optimizer) 中使用的参数和传统的动量 (momentum) 概念都不同。
        在数学上，这里运行统计的更新规则是 :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t` ，
        其中 :math:`\hat{x}` 是估计的统计量，:math:`x_t` 是新的观察值。

    因为 Batch Normalization 是在维度 `C` 上完成的，并在 `(N, D, H, W)` 切片上计算统计数据，所以学术上普遍称之为 Temporal Batch Normalization 。

    参数：
        - **num_features** : 来自大小为 :math:`(N, C, D, H, W)` 的预期输入的 :math:`C` 
        - **eps** : 为数值稳定性而添加到分母的值。默认： 1e-5
        - **momentum** : 用于 :attr:`running_mean` 和 :attr:`running_var` 计算的值。可以设置为 ``None`` 以计算累积移动平均。默认： 0.1
        - **affine** (bool, 可选): 如果为 ``True`` ，则该模块具有可学习的仿射参数。默认： ``True``
        - **track_running_stats** (bool, 可选): 如果为 ``True`` ，则此模块跟踪运行均值和方差。如果为 ``False`` ，此模块不跟踪此类统计信息，同时初始化统计缓冲区 :attr:`running_mean` 和 :attr:`running_var` 为 ``None`` 。当这些缓冲区为 ``None`` 时， 此模块在训练和评估模式中始终使用 batch statistics 。默认： ``True`` 

    形状：
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` （与输入形状相同）

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> x = flow.randn(3, 2, 5, 8, 4).to("cuda") # 目前 GPU 中支持 FusedBatchNorm 。
        >>> m = flow.nn.FusedBatchNorm3d(num_features=2, eps=1e-5, momentum=0.1).to("cuda")
        >>> y = m(x, addend=None)

    """
)

reset_docstr(
    oneflow.nn.GELU,
    r"""
    Gelu 激活算子。

    公式为：

    .. math::
        out = 0.5 * x * (1 + tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715x^{3})))

    参数：
        **x** (oneflow.tensor): 输入张量

    返回类型：
         oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([-0.5, 0, 0.5], dtype=flow.float32)
        >>> gelu = flow.nn.GELU()

        >>> out = gelu(input)
        >>> out
        tensor([-0.1543,  0.0000,  0.3457], dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.nn.GLU,
    r"""GLU(dim=-1)
    
    GLU 激活算子。

    参数：
        - **input** (Tensor, float): 输入张量
        - **dim** (int, 可选的): 分割输入的维度。默认：-1

    形状：
        - Input: :math:`(\ast_1, N, \ast_2)` 其中 `*` 表示任意数量的额外维度
        - Output: :math:`(\ast_1, M, \ast_2)` 其中 :math:`M=N/2`

    公式为：
    
    .. math::  

        GLU(input) = GLU(a, b) = a \otimes sigmoid(b)

    .. note::
        其中输入沿 :attr:`dim` 分成 a 和 b ，⊗ 是矩阵之间的元素积。

    示例：
    
    .. code-block:: python
    
        >>> import oneflow as flow
        >>> import oneflow.nn as nn
        >>> m = nn.GLU()
        >>> x = flow.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=flow.float32)
        >>> y = m(x)
        >>> y
        tensor([[0.9526, 1.9640],
                [4.9954, 5.9980]], dtype=oneflow.float32)
    
    
    """
)

reset_docstr(
    oneflow.nn.GroupNorm,
    r"""GroupNorm(num_groups: int, num_channels: int, eps: float = 1e-05, affine: bool = True)
    
    此接口与 PyTorch 对其，可参考以下文档：
    https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html

    对小批量输入应用组归一化 (Group Normalization) 的行为按论文 <https://arxiv.org/abs/1803.08494>`__ 中所述。

    公式为：

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    输入通道被分成 :attr:`num_groups` 个组，每组包含 ``num_channels / num_groups`` 个通道。
    每个组的平均值和标准差分开计算。如果 :attr:`affine` 为 ``True`` ，则
    :math:`\gamma` 和 :math:`\beta` 是大小为 :attr:`num_channels` 的可学习逐通道仿射变换参数向量
    (learnable per-channel affine transform parameter vectors)。

    通过有偏估计器 (biased estimator) 计算标准差，相当于 `torch.var(input, unbiased=False)` 。

    该层在训练和评估模式下都使用从输入计算的统计数据。

    参数：
        - **num_groups** (int): 将通道分成的组数
        - **num_channels** (int): 输入中预期的通道数
        - **eps** (float, 可选): 为数值稳定性而添加到分母的值。默认：1e-5
        - **affine** (bool, 可选): 如果为 ``True`` ，该模块具有可学习逐通道仿射变换参数，并初始化为 1 （对于权重）和 0（对于偏差）。默认： ``True`` 

    形状：
        - Input: :math:`(N, C, *)` 其中 :math:`C=\text{num_channels}`
        - Output: :math:`(N, C, *)` （与输入形状相同）

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.randn(20, 6, 10, 10)
        >>> # 将 6 个通道分成 3 组
        >>> m = flow.nn.GroupNorm(3, 6)
        >>> # 将6个通道分成6组（相当于InstanceNorm）
        >>> m = flow.nn.GroupNorm(6, 6)
        >>> # 将所有 6 个通道放在一个组中（相当于 LayerNorm）
        >>> m = flow.nn.GroupNorm(1, 6)
        >>> # 激活模块
        >>> output = m(input)
    
    """
)

reset_docstr(
    oneflow.nn.Hardsigmoid,
    r"""Hardsigmoid(inplace=False)
    
    应用逐元素公式：

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{ if } x \le -3  \\
            1 & \text{ if } x \ge +3 \\
            \frac{x}{6} + \frac{1}{2} & \text{ otherwise } \\
        \end{cases}

    参数：
        - **inplace** (bool): 是否进行 in-place 操作。默认： ``False``

    形状：
        - **Input** : :math:`(N, *)` 其中 `*` 表示任意数量的额外维度
        - **Output** : :math:`(N, *)`, 与输入相同的形状

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([-0.5, 0, 0.5], dtype=flow.float32)
        >>> hardsigmoid = flow.nn.Hardsigmoid()

        >>> out = hardsigmoid(input)
        >>> out
        tensor([0.4167, 0.5000, 0.5833], dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.nn.Hardswish,
    r"""Hardswish(inplace=False)
    
    如论文 `Searching for MobileNetV3`_ 中所述，逐元素应用 Hardswish 函数。

    公式为：

    .. math::
        \text{Hardswish}(x) = \begin{cases}
            0 & \text{ if } x \le -3  \\
            x & \text{ if } x \ge +3 \\
            x*(x+3)/6 & \text{ otherwise } \\
        \end{cases}

    参数：
        - **inplace** (bool, 可选): 是否执行 in-place 操作。默认： ``False`` 

    形状：
        - **Input** : :math:`(N, *)` 其中 `*` 表示任意数量的额外维度
        - **Output** : :math:`(N, *)` 与输入形状相同

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([-0.5, 0, 0.5], dtype=flow.float32)
        >>> hardswish = flow.nn.Hardswish()

        >>> out = hardswish(input)
        >>> out
        tensor([-0.2083,  0.0000,  0.2917], dtype=oneflow.float32)

    .. _`Searching for MobileNetV3`:
        https://arxiv.org/abs/1905.02244
    
    
    """
)

reset_docstr(
    oneflow.nn.Hardtanh,
    r"""Hardtanh(min_val=-1, max_val=1, inplace=False, min_value=None, max_value=None)

    逐元素应用 HardTanh 函数。

    公式为：

    .. math::
        \text{HardTanh}(x) = \begin{cases}
            1 & \text{ if } x > 1 \\
            -1 & \text{ if } x < -1 \\
            x & \text{ otherwise } \\
        \end{cases}

    可以使用参数 :attr:`min_val` 和 :attr:`max_val` 调整线性区域的范围 :math:`[-1, 1]`  。

    参数：
        - **min_val** (float): 线性区域范围的最小值。默认：-1
        - **max_val** (float): 线性区域范围的最大值。默认：1
        - **inplace** (bool): 是否执行 in-place 操作。默认： `False`

    关键词参数： :attr:`min_value` 和 :attr:`max_value` 已被弃用，由 :attr:`min_val` 和 :attr:`max_val` 替代。

    形状：
        - **Input** : :math:`(N, *)` 其中 `*` 表示任意数量的额外维度
        - **Output** : :math:`(N, *)` 与输入形状相同

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> m = flow.nn.Hardtanh()
        >>> x = flow.tensor([0.2, 0.3, 3.0, 4.0],dtype=flow.float32)
        >>> out = m(x)
        >>> out
        tensor([0.2000, 0.3000, 1.0000, 1.0000], dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.nn.Identity,
    r"""
    对参数不敏感的占位符标识运算符。

    示例：
        - **args** : 任何参数（未使用）
        - **kwargs** : 任何关键词参数（未使用）

    示例：

    .. code-block:: python

        import oneflow as flow

        m = flow.nn.Identity()
        input = flow.rand(2, 3, 4, 5)

        output = m(input)

        # output = input

    """
)

reset_docstr(
    oneflow.nn.InstanceNorm1d,
    r"""InstanceNorm1d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)

    此接口与 PyTorch 一致。可以在 https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm1d.html 参考相关文档。

    将实例归一化 (Instance Normalization) 应用于 3D 输入（具有可选附加通道维度的小批量 1D 输入），行为如论文
    `Instance Normalization: The Missing Ingredient for Fast Stylization
    <https://arxiv.org/abs/1607.08022>`__ 所述。

    公式为：

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    逐维度小批量单独计算均值和标准差。
    如果 :attr:`affine` 为 ``True`` ， :math:`\gamma` 和 :math:`\beta` 是大小为 `C` 的可学习参数向量（其中 `C` 是输入大小）。
    标准差是通过有偏估计器 (biased estimator) 计算的，相当于 `torch.var(input, unbiased=False)` 。

    默认情况下，该层在训练和评估模式下都使用从输入数据计算的实例统计信息。

    如果 :attr:`track_running_stats` 为 ``True`` ，在训练期间，该层会不断计算均值和方差的估计值，
    然后在评估期间将其用于归一化 (normalization) 。运行期间 :attr:`momentum` 为 0.1 。

    .. note::
        参数 :attr:`momentum` 与优化器 (optimizer) 中使用的参数和传统的动量 (momentum) 概念都不同。
        在数学上，这里运行统计的更新规则是 :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t` ，
        其中 :math:`\hat{x}` 是估计的统计量，:math:`x_t` 是新的观察值。

    .. note::
        尽管 :class:`InstanceNorm1d` 与 :class:`LayerNorm` 非常相似，但有一些细微的区别。 
        :class:`InstanceNorm1d` 应用于多维时间序列等通道数据的每个通道，但 :class:`LayerNorm` 
        通常应用于整个样本，并且经常应用于 NLP 任务。此外， :class:`LayerNorm` 应用逐元素仿射变换，
        而 :class:`InstanceNorm1d` 通常不应用仿射变换。

    参数：
        - **num_features** (int): 来自大小为 :math:`(N, C, L)` 的预期输入的 :math:`C` 或者来自大小为 :math:`(N, L)` 的预期输入的 :math:`L` 
        - **eps** (float): 为数值稳定性而添加到分母的值。默认：1e-5
        - **momentum** (float): 用于计算 running_mean 和 running_var 。默认：0.1
        - **affine** (bool): 如果为 ``True`` ，该模块具有可学习的仿射参数 (learnable affine parameters) ，初始化方式与批量标准化 (batch normalization) 相同。默认： ``False``
        - **track_running_stats** (bool): 如果为 ``True`` ，该模块记录运行均值和方差，如果为 ``True`` ，该模块不记录运行均值和方差，并且始终在训练和评估模式下都使用批处理该类统计信息。默认： ``False``

    形状：
        - **Input** : :math:`(N, C, L)`
        - **Output** : :math:`(N, C, L)` （与输入相同）

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> # 没有可学习的参数
        >>> m = flow.nn.InstanceNorm1d(100)
        >>> # 有可学习的参数
        >>> m = flow.nn.InstanceNorm1d(100, affine=True)
        >>> x = flow.randn(20, 100, 40)
        >>> output = m(x)

    """
)

reset_docstr(
    oneflow.nn.InstanceNorm2d,
    r"""InstanceNorm2d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)

    此接口与 PyTorch 一致。可以在 https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm1d.html 参考相关文档。

    将实例归一化 (Instance Normalization) 应用于 4D 输入（具有可选附加通道维度的小批量 2D 输入），行为如论文
    `Instance Normalization: The Missing Ingredient for Fast Stylization
    <https://arxiv.org/abs/1607.08022>`__ 所述。

    公式为：

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    逐维度小批量单独计算均值和标准差。
    如果 :attr:`affine` 为 ``True`` ， :math:`\gamma` 和 :math:`\beta` 是大小为 `C` 的可学习参数向量（其中 `C` 是输入大小）。
    标准差是通过有偏估计器 (biased estimator) 计算的，相当于 `torch.var(input, unbiased=False)` 。

    默认情况下，该层在训练和评估模式下都使用从输入数据计算的实例统计信息。

    如果 :attr:`track_running_stats` 为 ``True`` ，在训练期间，该层会不断计算均值和方差的估计值，
    然后在评估期间将其用于归一化 (normalization) 。运行期间 :attr:`momentum` 为 0.1 。

    .. note::
        参数 :attr:`momentum` 与优化器 (optimizer) 中使用的参数和传统的动量 (momentum) 概念都不同。
        在数学上，这里运行统计的更新规则是 :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t` ，
        其中 :math:`\hat{x}` 是估计的统计量，:math:`x_t` 是新的观察值。

    .. note::
        尽管 :class:`InstanceNorm2d` 与 :class:`LayerNorm` 非常相似，但有一些细微的区别。 
        :class:`InstanceNorm2d` 应用RGB图像等通道数据的每个通道，但 :class:`LayerNorm` 
        通常应用于整个样本，并且经常应用于 NLP 任务。此外， :class:`LayerNorm` 应用逐元素仿射变换，
        而 :class:`InstanceNorm2d` 通常不应用仿射变换。

    参数：
        - **num_features** (int): 来自大小为 :math:`(N, C, H, W)` 的预期输入的 :math:`C` 
        - **eps** (float): 为数值稳定性而添加到分母的值。默认：1e-5
        - **momentum** (float): 用于计算 running_mean 和 running_var 。默认：0.1
        - **affine** (bool): 如果为 ``True`` ，该模块具有可学习的仿射参数 (learnable affine parameters) ，初始化方式与批量标准化 (batch normalization) 相同。默认： ``False``
        - **track_running_stats** (bool): 如果为 ``True`` ，该模块记录运行均值和方差，如果为 ``True`` ，该模块不记录运行均值和方差，并且始终在训练和评估模式下都使用批处理该类统计信息。默认： ``False``
        
    形状：
        - **Input** : :math:`(N, C, H, W)`
        - **Output** : :math:`(N, C, H, W)` （与输入相同）

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> # 没有可学习的参数
        >>> m = flow.nn.InstanceNorm1d(100)
        >>> # 有可学习的参数
        >>> m = flow.nn.InstanceNorm1d(100, affine=True)
        >>> x = flow.randn(20, 100, 40)
        >>> output = m(x)

    """
)

reset_docstr(
    oneflow.nn.InstanceNorm3d,
    r"""InstanceNorm3d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)

    此接口与 PyTorch 一致。可以在 https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm1d.html 参考相关文档。

    将实例归一化 (Instance Normalization) 应用于 5D 输入（具有可选附加通道维度的小批量 3D 输入），行为如论文
    `Instance Normalization: The Missing Ingredient for Fast Stylization
    <https://arxiv.org/abs/1607.08022>`__ 所述。

    公式为：

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    逐维度小批量单独计算均值和标准差。
    如果 :attr:`affine` 为 ``True`` ， :math:`\gamma` 和 :math:`\beta` 是大小为 `C` 的可学习参数向量（其中 `C` 是输入大小）。
    标准差是通过有偏估计器 (biased estimator) 计算的，相当于 `torch.var(input, unbiased=False)` 。

    默认情况下，该层在训练和评估模式下都使用从输入数据计算的实例统计信息。

    如果 :attr:`track_running_stats` 为 ``True`` ，在训练期间，该层会不断计算均值和方差的估计值，
    然后在评估期间将其用于归一化 (normalization) 。运行期间 :attr:`momentum` 为 0.1 。

    .. note::
        参数 :attr:`momentum` 与优化器 (optimizer) 中使用的参数和传统的动量 (momentum) 概念都不同。
        在数学上，这里运行统计的更新规则是 :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t` ，
        其中 :math:`\hat{x}` 是估计的统计量，:math:`x_t` 是新的观察值。

    .. note::
        尽管 :class:`InstanceNorm3d` 与 :class:`LayerNorm` 非常相似，但有一些细微的区别。 
        :class:`InstanceNorm3d` 应用于具有RGB颜色的3D模型等通道数据的每个通道，但 :class:`LayerNorm` 
        通常应用于整个样本，并且经常应用于 NLP 任务。此外， :class:`LayerNorm` 应用逐元素仿射变换，
        而 :class:`InstanceNorm3d` 通常不应用仿射变换。

    参数：
        - **num_features** (int): 来自大小为 :math:`(N, C, D, H, W)` 的预期输入的 :math:`C` 
        - **eps** (float): 为数值稳定性而添加到分母的值。默认：1e-5
        - **momentum** (float): 用于计算 running_mean 和 running_var 。默认：0.1
        - **affine** (bool): 如果为 ``True`` ，该模块具有可学习的仿射参数 (learnable affine parameters) ，初始化方式与批量标准化 (batch normalization) 相同。默认： ``False``
        - **track_running_stats** (bool): 如果为 ``True`` ，该模块记录运行均值和方差，如果为 ``True`` ，该模块不记录运行均值和方差，并且始终在训练和评估模式下都使用批处理该类统计信息。默认： ``False``
        
    形状：
        - **Input** : :math:`(N, C, D, H, W)`
        - **Output** : :math:`(N, C, D, H, W)` （与输入相同）

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> # 没有可学习的参数
        >>> m = flow.nn.InstanceNorm3d(100)
        >>> # 有可学习的参数
        >>> m = flow.nn.InstanceNorm3d(100, affine=True)
        >>> x = flow.randn(20, 100, 35, 45, 10)
        >>> output = m(x)

    """
)
