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

    For example:

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
        - **start_dim** (int): first dim to flatten (default = 1).
        - **end_dim** (int): last dim to flatten (default = -1).
    
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
    
    在 2D 或 3D 输入上应用 Fused Batch Normalization，公式为：
    
    .. math:: 

        out = ReLU(BatchNorm(input) + addend)

    Batch Normalization 的公式为：

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\\gamma` and :math:`\\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\\gamma` are set
    to 1 and the elements of :math:`\\beta` are set to 0. The standard-deviation is calculated
    via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\\hat{x}_\\text{new} = (1 - \\text{momentum}) \\times \\hat{x} + \\text{momentum} \\times x_t`,
        where :math:`\\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``

    Shape:
        - **Input** : :math:`(N, C)` or :math:`(N, C, L)`
        - **Output** : :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> x = flow.randn(20, 100).to("cuda") # FusedBatchNorm support in GPU currently. 
        >>> m = flow.nn.FusedBatchNorm1d(num_features=100, eps=1e-5, momentum=0.1).to("cuda")
        >>> y = m(x, addend=None)

    """
)
