import oneflow
from docreset import reset_docstr

## oneflow.nn.AdaptiveAvgPool1d(output_size: Union[int, Tuple[int]])
## oneflow.nn.AdaptiveAvgPool2d(output_size)
## oneflow.nn.AdaptiveAvgPool3d(output_size)
## oneflow.nn.AvgPool1d(kernel_size: Union[int, Tuple[int, int]], stride: Optional[Union[int, Tuple[int, int]]] = None, padding: Union[int, Tuple[int, int]] = 0, ceil_mode: bool = False, count_include_pad: bool = True)
## oneflow.nn.AvgPool2d(kernel_size: Union[int, Tuple[int, int]], stride: Optional[Union[int, Tuple[int, int]]] = None, padding: Union[int, Tuple[int, int]] = 0, ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: int = 0)
## oneflow.nn.AvgPool3d(kernel_size: Union[int, Tuple[int, int, int]], stride: Optional[Union[int, Tuple[int, int, int]]] = None, padding: Union[int, Tuple[int, int, int]] = 0, ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: int = 0)
## oneflow.nn.BCELoss(weight: Optional[oneflow._oneflow_internal.Tensor] = None, reduction: str = 'mean')
## oneflow.nn.BCEWithLogitsLoss(weight: Optional[oneflow._oneflow_internal.Tensor] = None, reduction: str = 'mean', pos_weight: Optional[oneflow._oneflow_internal.Tensor] = None)
## oneflow.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
## oneflow.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
## oneflow.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
## oneflow.nn.CELU(alpha: float = 1.0, inplace: bool = False)
#### oneflow.nn.COCOReader


reset_docstr(
    oneflow.nn.CTCLoss,
    r"""CTCLoss(blank=0, reduction='mean', zero_infinity=False)

    计算 CTC(Connectionist Temporal Classification) 损失。
    
    此接口与 PyTorch 一致。文档参考自：https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html#torch.nn.CTCLoss 。

    计算连续且未分段的时间序列和 :attr:`target` 序列之间的损失。CTCLoss 对 :attr:`input` 与 :attr:`target` 可能对齐的概率求和，产生一个相对于每个 :attr:`input` 节点可微的损失值。
    假定 :attr:`input` 与 :attr:`target` 的对齐为“多对一”，这限制了 :attr:`target` 序列的长度，即 :attr:`target` :math:`\leq` :attr:`input`。
    
    参数：
        - **blank** (int, 可选的): 空白标签。默认值为 :math:`0`。
        - **reduction** (string, 可选的): 指定应用于输出的 reduction：``'none'`` | ``'mean'`` | ``'sum'``. ``'none'`` ：不进行 reduction；``'mean'`` ：输出损失将除以目标长度，然后取该批次的平均值。默认值为： ``'mean'``。
        - **zero_infinity** (bool, 可选的): 是否将无限损失和相关梯度归零。默认值为： ``False``。
            无限损失主要发生在 :attr:`inputs` 太短而无法与 :attr:`target` 对齐时。

    形状：
        - **Log_probs** : 形状为 :math:`(T, N, C)` 的张量且 :math:`T = \text{input length}`  、 :math:`N = \text{batch size}` 、 :math:`C = \text{number of classes (including blank)}`。
        - **Targets** : 形状为 :math:`(N, S)` 或 :math:`(\operatorname{sum}(\text{target_lengths}))` 的张量，其中 :math:`N = \text{batch size}` 、 :math:`S = \text{max target length, if shape is } (N, S)`。
          它代表 :attr:`target` 序列。 :attr:`target` 序列中的每个元素都是一个 class 索引。并且 :attr:`target` 索引不能为空（默认值为 0）。在 :math:`(N, S)` 形式中，:attr:`target` 被填充到最长序列的长度并堆叠。
          在 :math:`(\operatorname{sum}(\text{target_lengths}))` 形式中，我们假定目标在 1 维内未填充和连接。
        - **Input_lengths** : 大小为 :math:`(N)` 的元组或张量，其中 :math:`N = \text{batch size}`。它表示 :attr:`inputs` 的长度（每个都必须 :math:`\leq T`）。假定序列被填充为相等长度，为每个序列指定长度以实现掩码。
        - **Target_lengths** : 大小为 :math:`(N)` 的元组或张量，其中 :math:`N = \text{batch size}`。它代表 :attr:`target` 的长度。若假定序列被填充为相等长度，为每个序列指定长度以实现掩码。若 :attr:`target` 形状是 :math:`(N,S)`，
          则 target_lengths 是每个目标序列的有效停止索引 :math:`s_n` ，这样每个目标序列都满足 ``target_n = targets[n,0:s_n]`` ，长度都必须 :math:`\leq S`。
          若目标是作为单个目标的串联的 1d 张量给出的，则 target_lengths 必须加起来为张量的总长度。

    参考文献：
        A. Graves et al.: Connectionist Temporal Classification:
        Labelling Unsegmented Sequence Data with Recurrent Neural Networks:
        https://www.cs.toronto.edu/~graves/icml_2006.pdf

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> log_probs = flow.tensor(
        ...    [
        ...        [[-1.1031, -0.7998, -1.5200], [-0.9808, -1.1363, -1.1908]],
        ...        [[-1.2258, -1.0665, -1.0153], [-1.1135, -1.2331, -0.9671]],
        ...        [[-1.3348, -0.6611, -1.5118], [-0.9823, -1.2355, -1.0941]],
        ...        [[-1.3850, -1.3273, -0.7247], [-0.8235, -1.4783, -1.0994]],
        ...        [[-0.9049, -0.8867, -1.6962], [-1.4938, -1.3630, -0.6547]],
        ...    ], dtype=flow.float32)
        >>> targets = flow.tensor([[1, 2, 2], [1, 2, 2]], dtype=flow.int32)
        >>> input_lengths = flow.tensor([5, 5], dtype=flow.int32)
        >>> target_lengths = flow.tensor([3, 3], dtype=flow.int32)
        >>> loss_mean = flow.nn.CTCLoss()
        >>> out = loss_mean(log_probs, targets, input_lengths, target_lengths)
        >>> out
        tensor(1.1376, dtype=oneflow.float32)
        >>> loss_sum = flow.nn.CTCLoss(blank=0, reduction="sum")
        >>> out = loss_sum(log_probs, targets, input_lengths, target_lengths)
        >>> out
        tensor(6.8257, dtype=oneflow.float32)

    """
)

#### oneflow.nn.CoinFlip

reset_docstr(
    oneflow.nn.CombinedMarginLoss,
    r"""CombinedMarginLoss(m1=1.0, m2=0.0, m3=0.0)

    以下操作在 InsightFace 中实现了 "margin_softmax" ：
    https://github.com/deepinsight/insightface/blob/master/recognition/arcface_mxnet/train.py
    InsightFace 中 margin_softmax 的实现是由多个算子组成的。
    我们将它们组合在一起以加快速度。

    参数：
        - **input** (oneflow.Tensor): 输入张量。
        - **label** (oneflow.Tensor): 数据类型为整数的标签。
        - **m1** (float): 损失参数 m1。
        - **m2** (float): 损失参数 m2。
        - **m3** (float): 损失参数 m3。

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> x = flow.tensor([[-0.7027179, 0.0230609], [-0.02721931, -0.16056311], [-0.4565852, -0.64471215]], dtype=flow.float32)
        >>> label = flow.tensor([0, 1, 1], dtype=flow.int32)
        >>> loss_func = flow.nn.CombinedMarginLoss(0.3, 0.5, 0.4)
        >>> out = loss_func(x, label)
        >>> out
        tensor([[-0.0423,  0.0231],
                [-0.0272,  0.1237],
                [-0.4566, -0.0204]], dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.nn.ConstantPad1d,
    r"""ConstantPad1d(padding, value=0)

    用常数值填充输入 tensor 边界。此接口与 PyTorch 一致，文档参考自： https://pytorch.org/docs/stable/generated/torch.nn.ConstantPad1d.html 。

    用 :func:`torch.nn.functional.pad()` 来进行 `N` 维填充。

    参数:
        - **padding** (int, list, tuple): 填充的大小。若数据类型为 `int` 则在两个边界中使用相同的填充。若是 2-`tuple` ，则 (:math:`\text{padding_left}`, :math:`\text{padding_right}`)。
        - **value** (int, float): 用于填充的常量值。默认值为 0。

    形状：
        - **Input** : :math:`(N, C, W_{in})`
        - **Output** : :math:`(N, C, W_{out})` ，其中

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.arange(8, dtype=flow.float32).reshape(2,2,2)
        >>> m = flow.nn.ConstantPad1d(padding=[1, 2], value=9.9999)
        >>> output = m(input)
        >>> output
        tensor([[[9.9999, 0.0000, 1.0000, 9.9999, 9.9999],
                 [9.9999, 2.0000, 3.0000, 9.9999, 9.9999]],
        <BLANKLINE>
                [[9.9999, 4.0000, 5.0000, 9.9999, 9.9999],
                 [9.9999, 6.0000, 7.0000, 9.9999, 9.9999]]], dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.nn.ConstantPad2d,
    r"""ConstantPad2d(padding, value=0)
    
    此接口与 PyTorch 一致。文档参考自： https://pytorch.org/docs/stable/generated/torch.nn.ConstantPad2d.html 。

    用 0 填充输入张量边界。用户可以通过设置参数 :attr:`paddings` 来设置填充量。

    参数：
        - **padding** (int 或 tuple): 填充的大小。若是 `int`，则在所有边界中使用相同的填充。若是 4-`tuple` ，则(:math:`\mathrm{padding_{left}}`, :math:`\mathrm{padding_{right}}`, :math:`\mathrm{padding_{top}}`, :math:`\mathrm{padding_{bottom}}`)。

    形状：
        - **Input** : :math:`(N, C, H_{in}, W_{in})`
        - **Output** : :math:`(N, C, H_{out}, W_{out})` ，其中

            :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

            :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> m1 = flow.nn.ZeroPad2d(2)
        >>> m2 = flow.nn.ZeroPad2d((1,2,2,0))
        >>> input = flow.arange(18, dtype=flow.float32).reshape((1, 2, 3, 3))
        >>> output = m1(input)
        >>> output.shape
        oneflow.Size([1, 2, 7, 7])
        >>> output
        tensor([[[[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  1.,  2.,  0.,  0.],
                  [ 0.,  0.,  3.,  4.,  5.,  0.,  0.],
                  [ 0.,  0.,  6.,  7.,  8.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0.,  0.,  0.]],
        <BLANKLINE>
                 [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  9., 10., 11.,  0.,  0.],
                  [ 0.,  0., 12., 13., 14.,  0.,  0.],
                  [ 0.,  0., 15., 16., 17.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0.,  0.,  0.]]]], dtype=oneflow.float32)
        >>> output = m2(input)
        >>> output
        tensor([[[[ 0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  1.,  2.,  0.,  0.],
                  [ 0.,  3.,  4.,  5.,  0.,  0.],
                  [ 0.,  6.,  7.,  8.,  0.,  0.]],
        <BLANKLINE>
                 [[ 0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  9., 10., 11.,  0.,  0.],
                  [ 0., 12., 13., 14.,  0.,  0.],
                  [ 0., 15., 16., 17.,  0.,  0.]]]], dtype=oneflow.float32)
    """
)

reset_docstr(
    oneflow.nn.ConstantPad3d,
    r"""ConstantPad3d(padding, value=0)

    用常数值填充输入 tensor 边界。此接口与 PyTorch 一致，文档参考自： https://pytorch.org/docs/stable/generated/torch.nn.ConstantPad3d.html 。

    用 :func:`torch.nn.functional.pad()` 来进行 `N` 维填充。

    参数:
        - **padding** (int, list, tuple): 填充的大小。若数据类型为 `int` 则在所有边界中使用相同的填充。若是 6-`tuple` ，则 ( :math:`\text{padding_left}` , :math:`\text{padding_right}` , :math:`\text{padding_top}` , :math:`\text{padding_bottom}` , :math:`\text{padding_front}` , :math:`\text{padding_back}` )。
        - **value** (int, float): 用于填充的常量值。默认值为 0。

    形状：
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` ，其中

          :math:`D_{out} = D_{in} + \text{padding_front} + \text{padding_back}`

          :math:`H_{out} = H_{in} + \text{padding_top} + \text{padding_bottom}`

          :math:`W_{out} = W_{in} + \text{padding_left} + \text{padding_right}`

    示例：
    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.arange(8, dtype=flow.int32).reshape(1,1,2,2,2)
        >>> m = flow.nn.ConstantPad3d(padding=1, value=9)
        >>> output = m(input)
        >>> output
        tensor([[[[[9, 9, 9, 9],
                   [9, 9, 9, 9],
                   [9, 9, 9, 9],
                   [9, 9, 9, 9]],
        <BLANKLINE>
                  [[9, 9, 9, 9],
                   [9, 0, 1, 9],
                   [9, 2, 3, 9],
                   [9, 9, 9, 9]],
        <BLANKLINE>
                  [[9, 9, 9, 9],
                   [9, 4, 5, 9],
                   [9, 6, 7, 9],
                   [9, 9, 9, 9]],
        <BLANKLINE>
                  [[9, 9, 9, 9],
                   [9, 9, 9, 9],
                   [9, 9, 9, 9],
                   [9, 9, 9, 9]]]]], dtype=oneflow.int32)
    """

)

reset_docstr(
    oneflow.nn.Conv1d,
    r"""Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
    
    此接口与 PyTorch 一致。文档参考自：https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html#conv1d 。
    
    对由多个平面组成的输入信号应用 1D 卷积。

    在最简单的情况下，大小为 :math:`(N, C_{\text{in}}, L)` 的输入层的输出值和输出 :math:`(N, C_{\text{out}}, L_{\text{out}})` 可以被准确的表述为：

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{\text{out}_j}, k)
        \star \text{input}(N_i, k)

    其中 :math:`\star` 为有效的 `cross-correlation`_ 运算符， :math:`N` 是批量大小， :math:`C` 表示通道数， 
    :math:`L` 是信号序列的长度。

    * :attr:`stride` 是控制互相关 (cross-correlation) 的步幅 (stride) 的单个数字或单元素元组。

    * :attr:`padding` 控制应用于输入的填充量。可以是 `string` {{'valid', 'same'}}
      或一个给出在两侧的隐式填充量的整数元组。

    * :attr:`dilation` 控制核心点 (kernel points) 之间的间距，也称为 `à trous algorithm`。`link`_ 中有 :attr:`dilation` 的可视化展示。

    Note:
        ``padding='valid'`` 等同于无填充。 ``padding='same'`` 填充输入，使输出具有与输入相同的形状。
        但是在这种情况下，不支持除了 1 以外的任何步幅 (stride) 值。

    参数：
        - **in_channels** (int): 输入图像的通道数。
        - **out_channels** (int): 卷积产生的通道数。
        - **kernel_size** (int 或者 tuple): 卷积核的大小。
        - **stride** (int 或者 tuple, 可选的): 卷积的步幅 (stride)。默认值为： 1。
        - **padding** (int, tuple 或者 str, 可选的): 添加到输入两侧的填充值。默认值为： 0。
        - **padding_mode** (string, 可选的): 默认值为： ``'zeros'``。
        - **dilation** (int 或者 tuple, 可选的): 核心的元素之间的间距。默认值为： 1。
        - **groups** (int, 可选的): 从输入通道到输出通道的 `blocked connections` 数。默认值为：1。
        - **bias** (bool, 可选的): 若为 ``True`` ，则向输出添加可学习的偏差。默认值为： ``True``。

    形状：
        - **Input** : :math:`(N, C_{in}, L_{in})`
        - **Output** : :math:`(N, C_{out}, L_{out})` ，其中

          .. math::
              L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Attributes:
        weight (Tensor): 形状为 :math:`(\text{out\_channels}, 
            \frac{\text{in\_channels}}{\text{groups}}, \text{kernel\_size})` 的模块可学习权重。
            这些权重的值是由公式 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 计算而来，其中
            :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}` 

        bias (Tensor):  形状为 (out_channels) 的模块可学习权重。若 :attr:`bias` 为 ``True`` ，
            则那么这些权重的值是由公式 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 计算而来，其中
            :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}`

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import oneflow.nn as nn
        
        >>> input = flow.randn(20, 16, 50)
        >>> m = nn.Conv1d(16, 33, 3, stride=2)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

)  

reset_docstr(
    oneflow.nn.Conv2d,
    r"""Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
    
    此接口与 PyTorch 一致。文档参考自： https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html#conv2d 。
    
    对由多个平面组成的输入信号应用 2D 卷积。

    在最简单的情况下，大小为 :math:`(N, C_{\text{in}}, H, W)` 的输入层的输出值和输出 
    :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})` 可以被准确的表述为：

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)

    其中 :math:`\star` 为有效的 2D `cross-correlation`_ 运算符， :math:`N` 是批量大小， :math:`C` 表示通道数，
    :math:`H` 是以像素为单位的输入平面的高度，和 :math:`W` 是以像素为单位的宽度。


    * :attr:`stride` 是控制互相关 (cross-correlation) 的步幅 (stride) 的单个数字或单元素元组。

    * :attr:`padding` 控制在输入每个维度两侧隐式填充 :attr:`padding` 个点。

    * :attr:`dilation` 控制核心点 (kernel points) 之间的间距，也称为 `à trous algorithm`。此操作很难描述，
      但是 `link`_ 很好的将 :attr:`dilation` 的作用可视化。

    * :attr:`groups` 控制输入和输出之间的连接。 :attr:`in_channels` 和 :attr:`out_channels` 都必须能被 :attr:`groups` 整除。
      例如，

        * 当 groups=1 时，所有输入都卷积到输出。

        * 当 groups=2 时，该操作等效于并排放置两个 conv 层，其中每个层检查一半的输入通道并产生一半的输出通道，然后将两者连接起来。

        * 当 groups= :attr:`in_channels` 时， 每个输入通道都与它自己的一组过滤器(大小为
          :math:`\frac{\text{out_channels}}{\text{in_channels}}`)进行卷积。

    参数 :attr:`kernel_size` 、 :attr:`stride` 、 :attr:`padding` 、 :attr:`dilation` 可以是：

        - 单个 ``int`` -- 在这种情况下，高度和宽度使用相同的值。
        - 一个由两个 int 组成的 ``tuple`` -- 在这种情况下，第一个 `int` 用于高度，第二个 `int` 用于宽度。

    Note:
        当 `groups == in_channels` 并且 `out_channels == K * in_channels` 时，其中 `K` 是一个正整数，这个操作被称为“深度卷积”。

        换句话说，对于大小为 :math:`(N, C_{in}, L_{in})` 的输入，可以使用参数 :math:`(C_\text{in}=C_\text{in}, C_\text{out}=C_\text{in} \times \text{K}, ..., \text{groups}=C_\text{in})` 
        执行具有深度乘数 `K` 的深度卷积。

    参数：
        - **in_channels** (int): 输入图像的通道数。
        - **out_channels** (int): 卷积产生的通道数。
        - **kernel_size** (int 或者 tuple): 卷积核的大小。
        - **stride** (int 或者 tuple, 可选的): 卷积的步幅 (stride)。默认值为： 1。
        - **padding** (int, tuple 或者 str, 可选的): 添加到输入两侧的填充值。默认值为： 0。
        - **padding_mode** (string, 可选的): 默认值为： ``'zeros'``。
        - **dilation** (int 或者 tuple, 可选的): 核心的元素之间的间距。默认值为： 1。
        - **groups** (int, 可选的): 从输入通道到输出通道的 `blocked connections` 数。默认值为：1。
        - **bias** (bool, 可选的): 若为 ``True`` ，则向输出添加可学习的偏差。默认值为：``True``。

    形状：
        - **Input** : :math:`(N, C_{in}, H_{in}, W_{in})`
        - **Output** : :math:`(N, C_{out}, H_{out}, W_{out})` ，其中

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): 形状为 :math:`(\text{out_channels}, \frac{\text{in_channels}}{\text{groups}},`
            :math:`\text{kernel_size[0]}, \text{kernel_size[1]})` 的模块可学习权重。
            这些权重的值是由公式 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 计算而来，其中
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel_size}[i]}`

        bias (Tensor):  形状为 (out_channels) 的模块可学习权重。若 :attr:`bias` 为 ``True`` ，
            则那么这些权重的值是由公式 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 计算而来，其中
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel_size}[i]}`

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import oneflow.nn as nn
        
        >>> input = flow.randn(20, 16, 50, 100)
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

)

reset_docstr(
    oneflow.nn.Conv3d,
    r"""Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

    此接口与 PyTorch 一致。文档参考自：https://pytorch.org/docs/master/generated/torch.nn.Conv3d.html#conv3 。

    对由多个平面组成的输入信号应用 3D 卷积。

    在最简单的情况下，大小为 :math:`(N, C_{in}, D, H, W)` 的输入层的输出值和输出。
    :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` 可以被准确的表述为：

    .. math::
        out(N_i, C_{out_j}) = bias(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} weight(C_{out_j}, k) \star input(N_i, k)

    其中 :math:`\star` 为有效的 3D `cross-correlation`_ 运算符。

    * :attr:`stride` 是控制互相关 (cross-correlation) 的步幅 (stride) 的单个数字或单元素元组。

    * :attr:`padding` 控制应用于输入的填充量。可以是 `string` {{'valid', 'same'}}
      或一个给出在两侧的隐式填充量的整数元组。

    * :attr:`dilation` 控制核心点 (kernel points) 之间的间距，也称为 `à trous algorithm`。此操作很难描述，
      但是 `link`_ 很好的将 :attr:`dilation` 的作用可视化。

    参数 :attr:`kernel_size` 、 :attr:`stride` 、 :attr:`padding` 、 :attr:`dilation` 可以是：

        - 单个 ``int`` -- 在这种情况下，长度、宽度和高度使用相同的值
        - 一个由两个 int 组成的 ``tuple`` -- 在这种情况下，第一个 `int` 用于长度，第二个 `int` 用于高度，第三个 `int` 用于宽度。

    Note:
        ``padding='valid'`` 等同于无填充。 ``padding='same'`` 填充输入，使输出具有与输入相同的形状。
        但是在这种情况下，不支持除了 1 以外的任何步幅 (stride) 值。

    参数：
        - **in_channels** (int): 输入图像的通道数。
        - **out_channels** (int): 卷积产生的通道数。
        - **kernel_size** (int 或者 tuple): 卷积核的大小。
        - **stride** (int 或者 tuple, 可选的): 卷积的步幅 (stride)。默认值为： 1。
        - **padding** (int, tuple 或者 str, 可选的): 添加到输入两侧的填充值。默认值为： 0。
        - **padding_mode** (string, 可选的): 默认值为： ``'zeros'``。
        - **dilation** (int 或者 tuple, 可选的): 核心的元素之间的间距。默认值为： 1。
        - **groups** (int, 可选的): 从输入通道到输出通道的 `blocked connections` 数。默认值为：1。
        - **bias** (bool, 可选的): 若为 ``True`` ，则向输出添加可学习的偏差。默认值为：``True``。
    
    形状：
        - **Input** : :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - **Output** : :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` ，其中

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0]
                    \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1]
                    \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2]
                    \times (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Attributes:
        weight (Tensor): 形状为 :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]}, \text{kernel\_size[2]})` 的模块可学习权重。
                         这些权重的值是由公式 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 计算而来，其中
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}` 

        bias (Tensor):  形状为 (out_channels) 的模块可学习权重。若 :attr:`bias` 为 ``True`` ，
                         则那么这些权重的值是由公式 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 计算而来，其中
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import oneflow.nn as nn

        >>> input = flow.randn(1, 2, 5, 5, 5)
        >>> m = nn.Conv3d(2, 4, kernel_size=3, stride=1)
        >>> output = m(input)
        
    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation
    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
)

reset_docstr(
    oneflow.nn.ConvTranspose1d,
    r"""ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')

    在由多个输入平面组成的输入图像上应用 1D 转置卷积算子。

    该 module 可以看作是 Conv1d 相对于其输入的梯度。它也称为分数步幅卷积或反卷积（尽管它实际上不是反卷积操作）。

    此 module 支持 TensorFloat32。

    * :attr:`stride` 控制互相关 (cross-correlation) 的步幅 (stride)。

    * :attr:`padding` 控制应用于输入两侧，点的数量为 ``dilation * (kernel_size - 1) - padding`` 的隐式 0 填充。
      更多细节请参考 ``note``。

    * :attr:`output_padding`  控制添加到输出形状一侧的大小。更多信息请参考 ``note``。

    * :attr:`dilation` 控制核心点 (kernel points) 之间的间距，也称为 `à trous algorithm`。此操作很难描述，
      但是 `link`_ 很好的将 :attr:`dilation` 的作用可视化。

    Note:
        :attr:`padding` 参数有效地将 ``dilation * (kernel_size - 1) - padding`` 个 0 填充到输入的两侧。
            设定此项的目的是当 :class:`~torch.nn.Conv1d` 与 :class:`~torch.nn.ConvTranspose1d` 用相同的参数初始化时，
            它们的输入和输出的形状是互逆的。然而，当 ``stride > 1`` 时， :class:`~torch.nn.Conv1d` 
            将多个输入形状映射到相同的输出形状。则使用 :attr:`output_padding` 有效地增加一侧的输出形状来解决这种歧义。
            请注意，:attr:`output_padding` 仅用于查找输出形状，但实际上并未填充输出。

    Note:
        在某些情况下，将 CUDA 后端与 CuDNN 一起使用时，此运算符可能会选择非确定性算法来提高性能。
            若此操作有不确定性，您可以尝试通过设置 ``torch.backends.cudnn.deterministic =
            True`` 来使操作具有确定性（可能以性能为代价）。
            背景请参阅有关随机性 (randomness)  的 note。

    参数：
        - **in_channels** (int): 输入图像的通道数
        - **out_channels** (int): 卷积产生的通道数
        - **kernel_size** (int 或 tuple): 卷积核的大小
        - **stride** (int 或 tuple, 可选的): 卷积的步幅 (stride)。默认值为： 1
        - **padding** (int 或 tuple, 可选的): 添加到输入每侧的 ``dilation * (kernel_size - 1) - padding`` 大小的 0 填充值。默认值为： 0
        - **output_padding** (int 或 tuple, 可选的): 添加到输出形状一侧的大小。默认值为：0
        - **groups** (int, 可选的): 从输入通道到输出通道的 `blocked connections` 数。默认值为：1
        - **bias** (bool, 可选的): 若为 ``True`` ，则向输出添加可学习的偏差。默认值为：``True``
        - **dilation** (int 或 tuple, 可选的): 核心的元素之间的间距。默认值为： 1

    形状：
        - **Input** : :math:`(N, C_{in}, L_{in})`
        - **Output** : :math:`(N, C_{out}, L_{out})` ，其中

          .. math::
              L_{out} = (L_{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{dilation}
                        \times (\text{kernel_size} - 1) + \text{output_padding} + 1

    Attributes:
        weight (Tensor): 形状为 :math:`(\text{in\_channels}, \frac{\text{out\_channels}}{\text{groups}},`
                         :math:`\text{kernel\\_size})` 的模块可学习权重。
                         这些权重的值是由公式 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 计算而来，其中
                         :math:`k = \frac{groups}{C_\text{out} * \text{kernel\_size}}`

        bias (Tensor):   形状为 (out_channels) 的模块可学习权重。若 :attr:`bias` 为 ``True`` ，
                         则那么这些权重的值是由公式 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 计算而来，其中
                         :math:`k = \frac{groups}{C_\text{out} * \text{kernel\_size}}`
    
    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
)

reset_docstr(
    oneflow.nn.ConvTranspose2d,
    r"""ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
    
    在由多个输入平面组成的输入图像上应用 2D 转置卷积算子。

    该 module 可以看作是 Conv2d 相对于其输入的梯度。它也称为分数步幅卷积或反卷积（尽管它实际上不是反卷积操作）。

    参数：
        - **in_channels** (int): 输入图像的通道数。
        - **out_channels** (int): 卷积产生的通道数。
        - **kernel_size** (int 或 tuple): 卷积核的大小。
        - **stride** (int 或 tuple, 可选的): 卷积的步幅 (stride)。默认值为： 1。
        - **padding** (int 或 tuple, 可选的): 添加到输入每侧的 ``dilation * (kernel_size - 1) - padding`` 大小的 0 填充值。默认值为： 0。
        - **output_padding** (int 或 tuple, 可选的): 添加到输出形状一侧的大小。默认值为：0。
        - **groups** (int, 可选的): 从输入通道到输出通道的 `blocked connections` 数。默认值为：1。
        - **bias** (bool, 可选的): 若为 ``True`` ，则向输出添加可学习的偏差。默认值为：``True``。
        - **dilation** (int 或 tuple, 可选的): 核心的元素之间的间距。默认值为： 1。

    形状：
        - **Input** : :math:`(N, C_{in}, H_{in}, W_{in})`
        - **Output** : :math:`(N, C_{out}, H_{out}, W_{out})` ，其中

        .. math::
              H_{out} = (H_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0] 

                        \times (\text{kernel_size}[0] - 1) + \text{output_padding}[0] + 1
        .. math::
              W_{out} = (W_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1]
              
                        \times (\text{kernel_size}[1] - 1) + \text{output_padding}[1] + 1

    Attributes:
        ConvTranspose2d.weight (Tensor): 形状为 :math:`(\text{in_channels}, \frac{\text{out_channels}}{\text{groups}},`
                         :math:`\text{kernel_size[0]}, \text{kernel_size[1]})` 的模块可学习权重。
                         这些权重的值是由公式 `\mathcal{U}(-\sqrt{k}, \sqrt{k})` 计算而来，其中
                         :math:`k = \frac{groups}{C_\text{out} * \prod_{i=0}^{1}\text{kernel_size}[i]}`

        ConvTranspose2d.bias (Tensor):   形状为 (out_channels) 的模块可学习权重。若 :attr:`bias` 为 ``True`` ，
                         则那么这些权重的值是由公式 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 计算而来，其中
                         :math:`k = \frac{groups}{C_\text{out} * \text{kernel\_size}}`


    示例：
    
    .. code-block:: python

        >>> import oneflow as flow
        >>> import oneflow.nn as nn
        
        >>> m = nn.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> m = m.to("cuda")
        >>> input = flow.randn(20, 16, 50, 100, device=flow.device("cuda"))
        >>> output = m(input)
        >>> output.size()
        oneflow.Size([20, 33, 93, 100])

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
)

reset_docstr(
    oneflow.nn.ConvTranspose3d,
    r"""ConvTranspose3d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')

    在由多个输入平面组成的输入图像上应用 3D 转置卷积算子。

    转置卷积算子将每个输入值逐元素乘以一个可学习的内核 (kernel) ，并对所有输入特征平面的输出求和。

    该 module 可以看作是 Conv3d 相对于其输入的梯度。它也称为分数步幅卷积或反卷积（尽管它实际上不是反卷积操作）。

    此 module 支持 TensorFloat32。

    * :attr:`stride` 控制互相关 (cross-correlation) 的步幅 (stride)。

    * :attr:`padding` 控制应用于输入两侧，点的数量为 ``dilation * (kernel_size - 1) - padding`` 的隐式 0 填充。
      更多细节请参考 ``note``。

    * :attr:`output_padding`  控制添加到输出形状一侧的大小。更多信息请参考 ``note``。

    * :attr:`dilation` 控制核心点 (kernel points) 之间的间距，也称为 `à trous algorithm`。此操作很难描述，
      但是 `link`_ 很好的将 :attr:`dilation` 的作用可视化。

    参数 :attr:`kernel_size` 、 :attr:`stride` 、 :attr:`padding` 、 :attr:`output_padding` 可以是以下形式：

        - 单个 ``int`` -- 在这种情况下，长度、高度和宽度的大小使用相同的值。
        - 一个由三个 int 组成的 ``tuple`` -- 在这种情况下，第一个 `int` 用于长度，第二个 `int` 表示高度，第三个 `int` 表示宽度。

    Note:
        :attr:`padding` 参数有效地将 ``dilation * (kernel_size - 1) - padding`` 个 0 填充到输入的两侧。
        设定此项的目的是当 :class:`~torch.nn.Conv3d` 与 :class:`~torch.nn.ConvTranspose3d` 用相同的参数初始化时，
        它们的输入和输出的形状是互逆的。然而，当 ``stride > 1`` 时， :class:`~torch.nn.Conv3d` 
        将多个输入形状映射到相同的输出形状。则使用 :attr:`output_padding` 有效地增加一侧的输出形状来解决这种歧义。
        请注意，:attr:`output_padding` 仅用于查找输出形状，但实际上并未填充输出。

    参数：
        - **in_channels** (int): 输入图像的通道数。
        - **out_channels** (int): 卷积产生的通道数。
        - **kernel_size** (int 或 tuple): 卷积核的大小。
        - **stride** (int 或 tuple, 可选的): 卷积的步幅 (stride)。默认值为： 1。
        - **padding** (int 或 tuple, 可选的): 添加到输入每侧的 ``dilation * (kernel_size - 1) - padding`` 大小的 0 填充值。默认值为： 0。
        - **output_padding** (int 或 tuple, 可选的): 添加到输出形状一侧的大小。默认值为：0。
        - **groups** (int, 可选的): 从输入通道到输出通道的 `blocked connections` 数。默认值为：1。
        - **bias** (bool, 可选的): 若为 ``True`` ，则向输出添加可学习的偏差。默认值为：``True``。
        - **dilation** (int 或 tuple, 可选的): 核心的元素之间的间距。默认值为： 1。

    形状：
        - **Input** : :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - **Output** : :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` ，其中

        .. math::
              D_{out} = (D_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]
                        \times (\text{kernel_size}[0] - 1) + \text{output_padding}[0] + 1
        .. math::
              H_{out} = (H_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1]
                        \times (\text{kernel_size}[1] - 1) + \text{output_padding}[1] + 1
        .. math::
              W_{out} = (W_{in} - 1) \times \text{stride}[2] - 2 \times \text{padding}[2] + \text{dilation}[2]
                        \times (\text{kernel_size}[2] - 1) + \text{output_padding}[2] + 1

    Attributes:
        weight (Tensor): 形状为 :math:`(\text{in_channels}, \frac{\text{out_channels}}{\text{groups}},`
                         :math:`\text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})` 的模块可学习权重。
                         这些权重的值是由公式 `\mathcal{U}(-\sqrt{k}, \sqrt{k})` 计算而来，其中
                         :math:`k = \frac{groups}{C_\text{out} * \prod_{i=0}^{2}\text{kernel_size}[i]}`

        bias (Tensor):   形状为 (out_channels) 的模块可学习权重。若 :attr:`bias` 为 ``True`` ，
                         则那么这些权重的值是由公式 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 计算而来，其中
                         :math:`k = \frac{groups}{C_\text{out} * \text{kernel\_size}}`


    示例：
    
    .. code-block:: python

        >>> import oneflow as flow
        >>> import oneflow.nn as nn
        >>> # With square kernels and equal stride 
        >>> m = nn.ConvTranspose3d(16, 33, 3, stride=2) 
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.ConvTranspose3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2)) 
        >>> input = flow.randn(20, 16, 10, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
)

#### oneflow.nn.CropMirrorNormalize

reset_docstr(
    oneflow.nn.CrossEntropyLoss,
    r"""CrossEntropyLoss(weight=None, ignore_index=100, reduction='mean')

    将类 :class:`~flow.nn.LogSoftmax` 和 :class:`~flow.nn.NLLLoss` 组合在一起。

    该类在使用 `C` 类训练分类问题时很有用。

    :attr:`input` 应包含每个类的原始的，非标准化分数。

    在 `K` 维下， :attr:`input` 的大小必须为 :math:`(minibatch, C)` 或 :math:`(minibatch, C, d_1, d_2, ..., d_K)` ，
    其中 :math:`K \geq 1` （见下文）。

    在此标准中，类的索引应在 :math:`[0, C-1]` 范围内并引作为大小为 `minibatch` 的 1D tensor 的 `target` ；

    该损失可以被描述为：

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    通过提供大小为 :math:`(minibatch, C, d_1, d_2, ..., d_K)` 的输入和适当形状的目标（其中 :math:`K \geq 1` ， :math:`K` 是维度数），
    此类也可用于更高维度的输入，例如 2D 图像（见下文）。

    参数：
        - **reduction** (string, 可选的): 指定应用于输出的 reduction（可以是 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值为 ``'mean'``）：
            - ``'none'`` :不进行简化；
            - ``'mean'`` :取输出的加权平均值；
            - ``'sum'`` :取输出的和。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor(
        ...    [[-0.1664078, -1.7256707, -0.14690138],
        ...        [-0.21474946, 0.53737473, 0.99684894],
        ...        [-1.135804, -0.50371903, 0.7645404]], dtype=flow.float32)
        >>> target = flow.tensor([0, 1, 2], dtype=flow.int32)
        >>> out = flow.nn.CrossEntropyLoss(reduction="none")(input, target)
        >>> out
        tensor([0.8020, 1.1167, 0.3583], dtype=oneflow.float32)
        >>> out_sum = flow.nn.CrossEntropyLoss(reduction="sum")(input, target)
        >>> out_sum
        tensor(2.2769, dtype=oneflow.float32)
        >>> out_mean = flow.nn.CrossEntropyLoss(reduction="mean")(input, target)
        >>> out_mean
        tensor(0.7590, dtype=oneflow.float32)

    """)

## oneflow.nn.Dropout(p: float = 0.5, inplace: bool = False, generator=None)
## oneflow.nn.ELU(alpha: float = 1.0, inplace: bool = False)

reset_docstr(
    oneflow.nn.LeakyReLU,
    """LeakyReLU(negative_slope: float = 0.01, inplace: bool = False)
    
    逐元素应用公式：

    .. math::
        \\text{LeakyRELU}(x) = \\begin{cases}
            x, & \\text{ if } x \\geq 0 \\\\
            \\text{negative_slope} \\times x, & \\text{ otherwise }
        \\end{cases}

    参数：
        negative_slope: 控制负斜率的角度。默认值为 1e-2。
        inplace: 可以选择就地执行操作。默认值为 ``False``。

    形状：
        - **Input**: :math:`(N, *)` ，其中 `*` 表示任意数量的附加维度。
        - **Output**: :math:`(N, *)` ，与输入的形状相同。

    示例：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> m = flow.nn.LeakyReLU(0.1)
        >>> arr = np.array([0.2, 0.3, 3.0, 4.0])
        >>> x = flow.Tensor(arr)
        >>> out = m(x)
        >>> out
        tensor([0.2000, 0.3000, 3.0000, 4.0000], dtype=oneflow.float32)
    """
)

reset_docstr(
    oneflow.nn.Linear,
    """Linear(in_features: int, out_features: int, bias: bool = True)
    
    对输入数据应用线性变换： :math:`y = xA^T + b`

    参数：
        - **in_features**: 每一个输入样本的大小。
        - **out_features**: 每一个输出样本的大小。
        - **bias**: 若设置为 ``False`` ，则该层不会学习附加偏差。默认值为 ``True``。

    形状：
        - Input: :math:`(N, *, H_{in})` ，其中 `*` 表示任意数量的附加维度，且 :math:`H_{in} = {in\\_features}`。
        - Output: :math:`(N, *, H_{out})` ，其中除了最后一个维度之外的所有维度都与输入的形状相同，且 :math:`H_{out} = {out\\_features}`。

    属性：
        - :attr:`weight`: 形状为 :math:`({out\\_features}, {in\\_features})` 的模块的可学习参数。这些值通过 :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` 初始化，其中 :math:`(k = 1 / {in\\_features})`
        - :attr:`bias`: 形状为 :math:`({out\\_features})` 的模块的可学习参数。若 :attr:`bias` = ``True`` ，则这些值通过 :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` 初始化，其中 :math:`(k = 1 / {in\\_features})`


    示例：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        

        >>> m = flow.nn.Linear(20, 30, False)
        >>> input = flow.Tensor(np.random.randn(128, 20))
        >>> output = m(input)
        >>> output.size()
        oneflow.Size([128, 30])

    """
)

reset_docstr(
    oneflow.nn.LogSigmoid,
    """逐元素应用公式：

    .. math::
        \\text{LogSigmoid}(x) = \\log\\left(\\frac{ 1 }{ 1 + \\exp(-x)}\\right)

    形状：
        - Input: :math:`(N, *)` ，其中 `*` 表示任意数量的附加维度。
        - Output: :math:`(N, *)` ，与输入的形状相同。

    示例：

    .. code-block:: python


        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> logsigmoid = flow.nn.LogSigmoid()

        >>> out = logsigmoid(input)
        >>> out
        tensor([-0.9741, -0.6931, -0.4741], dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.nn.LogSoftmax,
    r"""LogSoftmax(dim: Optional[int] = None)
    
    对一个 n 维输入张量应用 LogSoftmax 公式。
    LogSoftmax 公式可以被简化为：

    .. math::
        \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right) = x_i - \log({ \sum_j \exp(x_j)})

    参数：
        - **dim** (int): 将沿其计算 LogSoftmax 的维度。

    形状：
        - Input: :math:`(N, *)` ，其中 `*` 表示任意数量的附加维度。
        - Output: :math:`(N, *)` ，与输入的形状相同。

    示例：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> m = flow.nn.LogSoftmax(dim=1)
        >>> x = flow.Tensor(
        ...    np.array(
        ...        [[ 0.4296, -1.1957,  2.5463],
        ...        [ 1.2552, -1.5747,  0.6923]]
        ...    )
        ... )
        >>> out = m(x)
        >>> out
        tensor([[-2.2513, -3.8766, -0.1346],
                [-0.4877, -3.3176, -1.0506]], dtype=oneflow.float32)
    """
)

reset_docstr(
    oneflow.nn.MarginRankingLoss,
    """MarginRankingLoss(margin: float = 0.0, reduction: str = 'mean')
    
    创造一个标准来衡量损失，给定输入 :math:`x1`, :math:`x2` ，两个 1D mini-batch `Tensors` ，以及一个带标签的 1D mini-batch tensor :math:`y` （包含 1 或 -1）。

    若 :math:`y = 1` 则假定第一个输入的 rank 比第二个输入更高， :math:`y = -1` 时反之亦然。

    小批量中每个样本的损失函数为：

    .. math::
        \\text{loss}(x1, x2, y) = \\max(0, -y * (x1 - x2) + \\text{margin})

    参数：
        - **margin** (float, optional): 默认值为 :math:`0`。
        - **reduction** (string, optional): 指定对输出应用的 reduction：``'none'`` | ``'mean'`` | ``'sum'``。``'none'`` ：不进行 reduction；``'mean'`` ：输出的和将会除以输出中的元素数量；``'sum'`` ：输出将被求和。默认值为 ``'mean'``。

    形状：
        - `x1` : :math:`(N, D)` ，其中 `N` 是批量大小， `D` 是样本大小。
        - `x2` : :math:`(N, D)` ，其中 `N` 是批量大小， `D` 是样本大小。
        - Target: :math:`(N)`
        - Output: 若 :attr:`reduction` = ``'none'`` ，那么输出为 :math:`(N)` ，否则为标量。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x1 = flow.tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=flow.float32)
        >>> x2 = flow.tensor(np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]]), dtype=flow.float32)
        >>> target = flow.tensor(np.array([[1, -1, 1],[-1, 1, -1], [1, 1, 1]]), dtype=flow.float32)
        >>> m = flow.nn.MarginRankingLoss(margin =1.0, reduction="none")
        >>> out = m(x1, x2, target)
        >>> out
        tensor([[2., 1., 0.],
                [3., 0., 5.],
                [0., 0., 0.]], dtype=oneflow.float32)

        >>> m = flow.nn.MarginRankingLoss(margin = 0.3, reduction="sum")
        >>> out = m(x1, x2, target)
        >>> out
        tensor(8.2000, dtype=oneflow.float32)

        >>> m = flow.nn.MarginRankingLoss(margin = 10, reduction="mean")
        >>> out = m(x1, x2, target)
        >>> out
        tensor(8.3333, dtype=oneflow.float32)


    """
)

reset_docstr(
    oneflow.nn.MaxPool1d,
    r"""MaxPool1d(kernel_size: Union[int, Tuple[int]], stride: Optional[Union[int, Tuple[int]]] = None, padding: Union[int, Tuple[int]] = 0, dilation: Union[int, Tuple[int]] = 1, return_indices: bool = False, ceil_mode: bool = False)
    
    此接口与 PyTorch 一致。文档参考自： https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html#torch.nn.MaxPool1d 。

    在一个由多个输入平面组成的输入信号上应用 1D max pooling。

    在最简单的情况下，若输入大小为 :math:`(N, C, L)` 和输出大小为 :math:`(N, C, L_{out})` ，则该层的输出值可以被准确描述为：

    .. math::
        out(N_i, C_j, k) = \max_{m=0, \ldots, \text{kernel\_size} - 1}
                input(N_i, C_j, stride \times k + m)

    若 :attr:`padding` 非负，则在输入的两侧使用最小值隐式填充，以填充点数。 :attr:`dilation` 是滑动窗口中元素之间的跨步。 `链接 <https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md>`__  有一个池化参数的可视化展示。

    Note:
        当 ceil_mode = True 且滑动窗口从左侧填充区域或输入中开始，则允许其越界。从右侧填充区域开始的滑动窗口将被忽略。

    参数：
        - **kernel_size**: 滑动窗口的大小，必须为正。
        - **stride**: 滑动窗口的步长，必须为正。默认值为 :attr:`kernel_size`。
        - **padding**: 两侧都用隐式的负无穷大填充，该值必须非负且不大于 kernel_size / 2。
        - **dilation**: 滑动窗口中元素之间的跨步，必须为正。
        - **return_indices**: 若设置为 ``True`` 则返回 argmax 以及最大值，在后续的 :class:`torch.nn.MaxUnpool1d` 中用到。
        - **ceil_mode**: 若设置为 ``True`` 则使用 `ceil` 而非 `floor` 来计算输出形状。这确保了输入张量中的每个元素都被滑动窗口覆盖。

    形状：
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, L_{out})` ，其中

          .. math::
              L_{out} = \left\lfloor \frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                    \times (\text{kernel_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    示例： 

    .. code-block:: python 

        import oneflow as flow 
        import numpy as np

        of_maxpool1d = flow.nn.MaxPool1d(kernel_size=3, padding=1, stride=1)
        x = flow.Tensor(np.random.randn(1, 4, 4))
        y = of_maxpool1d(x)
        y.shape 
        oneflow.Size([1, 4, 4])

    """
)

reset_docstr(
    oneflow.nn.MaxPool2d,
    r"""MaxPool2d(kernel_size: Union[int, Tuple[int, int]], stride: Optional[Union[int, Tuple[int, int]]] = None, padding: Union[int, Tuple[int, int]] = 0, dilation: Union[int, Tuple[int, int]] = 1, return_indices: bool = False, ceil_mode: bool = False)
    
    此接口与 PyTorch 一致。文档参考自： https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d 。
    
    在一个由多个输入平面组成的输入信号上应用 2D max pooling。

    在最简单的情况下，若输入为 :math:`(N, C, H, W)` ，输出为 :math:`(N, C, H_{out}, W_{out})` 且 :attr:`kernel_size` 为 :math:`(kH, kW)` ，则该层的输出值可以被准确描述为：

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
        \end{aligned}

    若 :attr:`padding` 非负，则在输入的两侧使用最小值隐式填充，以填充点数。 :attr:`dilation` 控制了核点之间的空间。
    这很难描述，但这个 `链接 <https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md>`__ 以可视化的方式说明了 :attr:`dilation` 在做什么。

    Note:
        若 ceil_mode == True 且滑动窗口从左侧填充区域或输入中开始，则允许滑动窗口越界。从右侧填充区域开始的滑动窗口将被忽略。

    参数 :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` 可以是：
        - 一个单独的 ``int`` -- 在这种情况下，高度和宽度维度使用相同的值。
        - 一个由两个 int 组成的 ``tuple`` -- 在这种情况下，第一个整数用于高度维度，第二个整数用于宽度维度。

    参数：
        - **kernel_size**: 窗口的最大尺寸。
        - **stride**: 窗口的滑动步长。默认值为 :attr:`kernel_size`。
        - **padding**: 要加在两侧的隐式最小填充值。
        - **dilation**: 控制窗口中元素步幅的参数。
        - **return_indices**: 若设置为 ``True`` ，则返回最大索引和输出，在后续的 :class:`torch.nn.MaxUnpool2d` 中用到。
        - **ceil_mode**: 若设置为 ``True`` 则使用 `ceil` 而非 `floor` 来计算输出形状。

    形状：
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` ，其中

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                    \times (\text{kernel_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor
          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor

    示例：

    .. code-block:: python

        import oneflow as flow 
        import numpy as np

        m = flow.nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        x = flow.Tensor(np.random.randn(1, 4, 4, 4))
        y = m(x)
        y.shape 
        oneflow.Size([1, 4, 4, 4])

    """
)

reset_docstr(
    oneflow.nn.MaxPool3d,
    r"""MaxPool3d(kernel_size: Union[int, Tuple[int, int, int]], stride: Optional[Union[int, Tuple[int, int, int]]] = None, padding: Union[int, Tuple[int, int, int]] = 0, dilation: Union[int, Tuple[int, int, int]] = 1, return_indices: bool = False, ceil_mode: bool = False)
    
    此接口与 PyTorch 一致。文档参考自： https://pytorch.org/docs/stable/generated/torch.nn.MaxPool3d.html#torch.nn.MaxPool3d 。

    在一个由多个输入平面组成的输入信号上应用 3D max pooling。

    在最简单的情况下，若输入为 :math:`(N, C, D, H, W)` ，输出为 :math:`(N, C, D_{out}, H_{out}, W_{out})` 且 :attr:`kernel_size` 为 :math:`(kD, kH, kW)` ，则该层的输出值可以被准确描述为：

    .. math::
        \begin{aligned}
            \text{out}(N_i, C_j, d, h, w) ={} & \max_{k=0, \ldots, kD-1} \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                              & \text{input}(N_i, C_j, \text{stride[0]} \times d + k,
                                                             \text{stride[1]} \times h + m, \text{stride[2]} \times w + n)
        \end{aligned}

    若 :attr:`padding` 非负，则在输入的两侧使用最小值隐式填充，以填充点数。 :attr:`dilation` 控制了核点之间的空间。
    这很难描述，但这个 `链接 <https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md>`__ 以可视化的方式说明了 :attr:`dilation` 在做什么。

    Note:
        若 ceil_mode == True 且滑动窗口从左侧填充区域或输入中开始，则允许滑动窗口越界。从右侧填充区域开始的滑动窗口将被忽略。

    参数 :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` 可以是：
        - 一个单独的 ``int`` -- 在这种情况下，深度、高度和宽度维度使用相同的值。
        - 一个由三个 int 组成的 ``tuple`` -- 在这种情况下，第一个整数用于深度维度，第二个整数用于高度维度，第三个整数用于宽度维度。

    参数：
        - **kernel_size**: 窗口的最大尺寸。
        - **stride**: 窗口的滑动步长。默认值为 :attr:`kernel_size`。
        - **padding**: 在三个边上都用隐式的最小值填充。
        - **dilation**: 控制窗口中元素步幅的参数。
        - **return_indices**: 若设置为 ``True`` ，则返回最大索引和输出，在后续的 :class:`torch.nn.MaxUnpool3d` 中用到。
        - **ceil_mode**: 若设置为 ``True`` 则使用 `ceil` 而非 `floor` 来计算输出形状。

    形状：
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` ，其中

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0] \times
                (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1] \times
                (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2] \times
                (\text{kernel_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    示例：

    .. code-block:: python

        import oneflow as flow 
        import numpy as np 

        of_maxpool3d = flow.nn.MaxPool3d(kernel_size=3, padding=1, stride=1)
        x = flow.Tensor(np.random.randn(1, 4, 4, 4, 4))
        y = of_maxpool3d(x)
        y.shape 
        oneflow.Size([1, 4, 4, 4, 4])

    """
)

reset_docstr(
    oneflow.nn.MinMaxObserver,
    """MinMaxObserver(quantization_formula: str = 'google', quantization_bit: int = 8, quantization_scheme: str = 'symmetric', per_layer_quantization: bool = True)
    
    计算输入张量的量化参数。

    首先计算输入张量的最大值和最小值：

    .. math::

        & max\\_value = max(input)

        & min\\_value = min(input)

    然后用以下等式计算 scale 和 zero_point ：

        若 quantization_scheme == "symmetric":

        .. math::

            & denom = 2^{quantization\\_to\\_bit - 1} - 1

            & scale = max(|max\\_value|,|min\\_value|) / denom

            & zero\\_point = 0

        若 quantization_scheme == "affine":

        .. math::

            & denom = 2^{quantization\\_to\\_bit} - 1

            & scale = (max\\_value - min\\_value) / denom

            & zero\\_point = -min\\_value / scale

    若 per_layer_quantization == False ，则 scale 和 zero_point 的形状为 (input.shape[0],)。

    参数：
        - **quantization_bit** (int): 量化输入为 uintX / intX ， X 的值在 [2, 8] 中，默认值为 8。
        - **quantization_scheme** (str): "symmetric" 或 "affine" ， 量化为有符号/无符号整数。 默认值为 "symmetric"。
        - **quantization_formula** (str): "google" or "cambricon"。
        - **per_layer_quantization** (bool): 若设置为 True ，则表示 per-layer ，否则为 per-channel。默认值为 True。

    返回值：
        Tuple[oneflow.Tensor, oneflow.Tensor]: 输入张量的 scale 和 zero_point

    示例：

    .. code-block:: python
        
        >>> import numpy as np
        >>> import oneflow as flow

        >>> weight = (np.random.random((2, 3, 4, 5)) - 0.5).astype(np.float32)
        
        >>> input_tensor = flow.tensor(
        ...    weight, dtype=flow.float32
        ... )
        
        >>> quantization_bit = 8
        >>> quantization_scheme = "symmetric"
        >>> quantization_formula = "google"
        >>> per_layer_quantization = True

        >>> min_max_observer = flow.nn.MinMaxObserver(quantization_formula=quantization_formula, quantization_bit=quantization_bit,
        ... quantization_scheme=quantization_scheme, per_layer_quantization=per_layer_quantization)

        >>> scale, zero_point = min_max_observer(
        ...    input_tensor, )

    """
)

#### oneflow.nn.ModuleDict
#### oneflow.nn.ModuleList


reset_docstr(
    oneflow.nn.MovingAverageMinMaxObserver,
    """MovingAverageMinMaxObserver(training: bool = False, quantization_formula: str = 'google', stop_update_after_iters: int = 0, quantization_bit: int = 8, quantization_scheme: str = 'symmetric', momentum: float = 0)
    
    根据输入张量的最小值和最大值的移动平均计算量化参数。

    首先计算输入张量的 moving\\_max 和 moving\\_min ：

        若 quantization_scheme == "symmetric":

        .. math::

            & moving\\_max = moving\\_max * momentum + |max(input)| * (1 - momentum)

            & moving\\_min = moving\\_max

        若 quantization_scheme == "affine":

        .. math::

            & moving\\_max = moving\\_max * momentum + max(input) * (1 - momentum)

            & moving\\_min = moving\\_min * momentum + min(input) * (1 - momentum)

    最小值和最大值的移动平均值被初始化为第一批输入 `Blob` 的最小值和最大值。

    然后用以下等式计算 scale 和 zero_point ：

        若 quantization_scheme == "symmetric":

        .. math::

            & denom = 2^{quantization\\_to\\_bit - 1} - 1

            & scale = moving\\_max / denom

            & zero\\_point = 0

        若 quantization_scheme == "affine":

        .. math::

            & denom = 2^{quantization\\_to\\_bit} - 1

            & scale = (moving\\_max - moving\\_min) / denom

            & zero\\_point = -moving\\_min / scale

    Note:
        ``current_train_step`` 可以直接被赋值给一个优化器（例如 SGD）

    参数：
        - **training** (bool): 模式是否处于训练状态，默认值为 False。
        - **quantization_bit** (int): 量化输入为 uintX / intX ， X 的值在 [2, 8] 中，默认值为 8。
        - **quantization_scheme** (str): "symmetric" 或 "affine" ， 量化为有符号/无符号整数。 默认值为 "symmetric"。
        - **quantization_formula** (str): "google" or "cambricon"。
        - **momentum** (float): 指数移动平均运算的平滑参数，默认值为 0.95。

    返回值：
        Tuple[oneflow.Tensor, oneflow.Tensor]: 输入张量的 scale 和 zero_point

    示例：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> weight = (np.random.random((2, 3, 4, 5)) - 0.5).astype(np.float32)
        
        >>> input_tensor = flow.tensor(
        ...    weight, dtype=flow.float32
        ... )

        >>> current_train_step_tensor = flow.tensor(
        ...   np.zeros((1,)).astype(np.float32),
        ...    dtype=flow.int64,
        ... )
        
        >>> momentum = 0.95
        >>> quantization_bit = 8
        >>> quantization_scheme = "symmetric"
        >>> quantization_formula = "google"

        >>> moving_average_min_max_observer = flow.nn.MovingAverageMinMaxObserver(training=True, quantization_formula=quantization_formula, 
        ...                                                                       stop_update_after_iters=1, quantization_bit=quantization_bit,
        ...                                                                       quantization_scheme=quantization_scheme, momentum=momentum,
        ...                                                                       )

        >>> (scale, zero_point) = moving_average_min_max_observer(
        ...    input_tensor,
        ...    current_train_step_tensor,
        ... )

    """
)

reset_docstr(
    oneflow.nn.NLLLoss,
    """
    
    负对数似然损失。用 `C` 类训练分类问题很有用。

    The `input` given through a forward call is expected to contain
    log-probabilities of each class. `input` has to be a Tensor of size either
    :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \\geq 1` for the `K`-dimensional case (described later).

    Obtaining log-probabilities in a neural network is easily achieved by
    adding a  `LogSoftmax`  layer in the last layer of your network.
    You may use `CrossEntropyLoss` instead, if you prefer not to add an extra
    layer.

    The `target` that this loss expects should be a class index in the range :math:`[0, C-1]`
    where `C = number of classes`;

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
        l_n = - w_{y_n} x_{n,y_n}, \\quad
        w_{c} = \\mathbb{1},

    where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight, and
    :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then

    .. math::
        \\ell(x, y) = \\begin{cases}
            \\sum_{n=1}^N \\frac{1}{N} l_n, &
            \\text{if reduction} = \\text{`mean';}\\\\
            \\sum_{n=1}^N l_n,  &
            \\text{if reduction} = \\text{`sum'.}
        \\end{cases}

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \\geq 1`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape
    (see below). In the case of images, it computes NLL loss per-pixel.

    参数：
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Default: ``'mean'``

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> input = flow.tensor(
        ... [[-0.1664078, -1.7256707, -0.14690138],
        ... [-0.21474946, 0.53737473, 0.99684894],
        ... [-1.135804, -0.50371903, 0.7645404]], dtype=flow.float32)
        >>> target = flow.tensor(np.array([0, 1, 2]), dtype=flow.int32)
        >>> m = flow.nn.NLLLoss(reduction="none")
        >>> out = m(input, target)
        >>> out
        tensor([ 0.1664, -0.5374, -0.7645], dtype=oneflow.float32)

        >>> m = flow.nn.NLLLoss(reduction="sum")
        >>> out = m(input, target)
        >>> out
        tensor(-1.1355, dtype=oneflow.float32)

        >>> m = flow.nn.NLLLoss(reduction="mean")
        >>> out = m(input, target)
        >>> out
        tensor(-0.3785, dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.nn.OFRecordBytesDecoder,
    r"""OFRecordBytesDecoder(blob_name: str, name: Optional[str] = None)
    
    此运算符将张量读取为字节，输出取决于下游任务，可能需要进一步的解码过程，比如 cv2.imdecode() 用于图像和解码，以及 decode("utf-8") 用于字符。
    
    参数：
        - **blob_name**: 目标特征的名称。
        - **name**: 图中此分量的名称。
        - **input**: 可能由 OFRecordReader 提供的张量。

    返回值：
        按字节编码后的张量

    示例：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> def example():
        ...      batch_size = 16
        ...      record_reader = flow.nn.OFRecordReader(
        ...         "dataset/",
        ...         batch_size=batch_size,
        ...         part_name_suffix_length=5,
        ...      )
        ...      val_record = record_reader()

        ...      bytesdecoder_img = flow.nn.OFRecordBytesDecoder("encoded")

        ...      image_bytes_batch = bytesdecoder_img(val_record)

        ...      image_bytes = image_bytes_batch.numpy()[0]
        ...      return image_bytes
        ... example()  # doctest: +SKIP
        array([255 216 255 ...  79 255 217], dtype=uint8)



    """
)

#### oneflow.nn.OFRecordImageDecoder
#### oneflow.nn.OFRecordImageDecoderRandomCrop
#### oneflow.nn.OFRecordRawDecoder
#### oneflow.nn.OFRecordReader

reset_docstr(
    oneflow.nn.PReLU,
    """PReLU(num_parameters: int = 1, init: float = 0.25, device=None, dtype=None)
    
    逐元素应用公式：

    .. math::
        PReLU(x) = \\max(0,x) + a * \\min(0,x)

    这里 :math:`a` 是一个可学习的参数。当不带参数调用时， `nn.PReLU()` 在所有输入通道中使用单个参数 :math:`a`。
    若调用 `nn.PReLU(nChannels)` ，为每个通道使用单独的 :math:`a`。

    .. note::
        为了获得良好的性能，在学习 :math:`a` 时不应使用权重衰减。

    .. note::
        通道维度是输入的第二维度。当输入维度不足 2 时，就不存在通道维度且通道数为 1。

    参数：
        - **num_parameters** (int): 需要学习的 :math:`a` 的数量尽管它将一个 int 数值作为输入，但只有两类值是合法的： 1 或输入的通道数。默认值为 1。
        - **init** (float): :math:`a` 的初始值。默认值为 0.25。

    形状：
        - Input: :math:`(N, *)` ，其中 `*` 表示任意数量的附加维度。
        - Output: :math:`(N, *)` ，与输入的形状相同。

    属性：
        - weight (Tensor): 形状 (:attr:`num_parameters`) 的可学习权重。

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> m = flow.nn.PReLU()
        >>> input = flow.tensor(np.asarray([[[[1, -2], [3, 4]]]]), dtype=flow.float32)
        >>> print(m(input).numpy())
        [[[[ 1.  -0.5]
           [ 3.   4. ]]]]

    """
)

#### oneflow.nn.Parameter
#### oneflow.nn.ParameterDict
#### oneflow.nn.ParameterList
#### oneflow.nn.PixelShuffle

reset_docstr(
    oneflow.nn.Quantization,
    """FakeQuantization(quantization_formula: str = 'google', quantization_bit: int = 8, quantization_scheme: str = 'symmetric')
    
    在推理时模拟量化操作。

    输出将被计算为：

        若 quantization_scheme == "symmetric":

        .. math::

            & quant\\_max = 2^{quantization\\_to\\_bit - 1} - 1

            & quant\\_min = -quant\\_max

            & clamp(round(x / scale), quant\\_min, quant\\_max)

        若 quantization_scheme == "affine":

        .. math::

            & quant\\_max = 2^{quantization\\_to\\_bit} - 1

            & quant\\_min = 0

            & (clamp(round(x / scale + zero\\_point), quant\\_min, quant\\_max) - zero\\_point)

    参数：
        - **quantization_bit** (int): 量化输入为 uintX / intX ， X 的值在 [2, 8] 中，默认值为 8。
        - **quantization_scheme** (str): "symmetric" 或 "affine" ， 量化为有符号/无符号整数。 默认值为 "symmetric"。
        - **quantization_formula** (str): "google" or "cambricon"。

    返回值：
        oneflow.Tensor: 经过量化操作后的输入张量。

    示例：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> weight = (np.random.random((2, 3, 4, 5)) - 0.5).astype(np.float32)
        
        >>> input_tensor = flow.tensor(
        ...    weight, dtype=flow.float32
        ... )
        
        >>> quantization_bit = 8
        >>> quantization_scheme = "symmetric"
        >>> quantization_formula = "google"
        >>> per_layer_quantization = True

        >>> min_max_observer = flow.nn.MinMaxObserver(quantization_formula=quantization_formula, quantization_bit=quantization_bit,
        ... quantization_scheme=quantization_scheme, per_layer_quantization=per_layer_quantization)
        >>> quantization = flow.nn.Quantization(quantization_formula=quantization_formula, quantization_bit=quantization_bit, 
        ... quantization_scheme=quantization_scheme)

        >>> scale, zero_point = min_max_observer(
        ...    input_tensor,
        ... )

        >>> output_tensor = quantization(
        ...    input_tensor,
        ...    scale,
        ...    zero_point,
        ... )

    """
)

## oneflow.nn.ReLU


reset_docstr(
    oneflow.nn.ReLU6,
    """ReLU6(inplace: bool = False)
    
    逐元素应用公式：

    .. math::

        \\text{Relu6}(x) = \\begin{cases}
            6 & \\text{ if } x > 6 \\\\
            0 & \\text{ if } x < 0 \\\\
            x & \\text{ otherwise } \\\\
        \\end{cases}

    参数：
        - **inplace**: 可以选择就地执行操作。默认值为 ``False``。

    形状：
        - Input: :math:`(N, *)` ，其中 `*` 表示任意数量的附加维度。
        - Output: :math:`(N, *)` ，与输入的形状相同。

    示例：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> relu6 = flow.nn.ReLU6()

        >>> out = relu6(input)
        >>> out
        tensor([0.0000, 0.0000, 0.5000], dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.nn.ReflectionPad2d,
    """ReflectionPad2d(padding: Union[int, Tuple[int, int, int, int]])
    此接口与 PyTorch 一致。文档参考自： https://pytorch.org/docs/stable/generated/torch.nn.ReflectionPad2d.html 。

    使用输入边界的反射来填充输入张量。

    参数：
        - **padding** (Union[int,tuple]): 填充范围的大小或边界。若输入是一个 int，那各个维度上都会填充同样大小的数据。若输入是一个四个元素的元组，那么使用 :math:`(\\text{padding}_{\\text{left}}, \\text{padding}_{\\text{right}}, \\text{padding}_{\\text{top}}, \\text{padding}_{\\text{bottom}} )`。

    返回值：
        Tensor: 返回一个新的张量，这是输入张量的反射填充的结果。

    形状：
        - Input: :math:`(N, C, H_{\\text{in}}, W_{\\text{in}})`
        - Output: :math:`(N, C, H_{\\text{out}}, W_{\\text{out}})` ，其中

          :math:`H_{\\text{out}} = H_{\\text{in}} + \\text{padding}_{\\text{top}} + \\text{padding}_{\\text{bottom}}`

          :math:`W_{\\text{out}} = W_{\\text{in}} + \\text{padding}_{\\text{left}} + \\text{padding}_{\\text{right}}`

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.tensor(np.arange(18).reshape((1, 2, 3, 3)).astype(np.float32))
        >>> m = flow.nn.ReflectionPad2d((2, 2, 1, 1))
        >>> out = m(input)
        >>> out
        tensor([[[[ 5.,  4.,  3.,  4.,  5.,  4.,  3.],
                  [ 2.,  1.,  0.,  1.,  2.,  1.,  0.],
                  [ 5.,  4.,  3.,  4.,  5.,  4.,  3.],
                  [ 8.,  7.,  6.,  7.,  8.,  7.,  6.],
                  [ 5.,  4.,  3.,  4.,  5.,  4.,  3.]],
        <BLANKLINE>
                 [[14., 13., 12., 13., 14., 13., 12.],
                  [11., 10.,  9., 10., 11., 10.,  9.],
                  [14., 13., 12., 13., 14., 13., 12.],
                  [17., 16., 15., 16., 17., 16., 15.],
                  [14., 13., 12., 13., 14., 13., 12.]]]], dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.nn.ReplicationPad2d,
    """ReplicationPad2d(padding: Union[int, Tuple[int, int, int, int]])
    
    此接口与 PyTorch 一致。文档参考自： https://pytorch.org/docs/stable/generated/torch.nn.ReplicationPad2d.html#replicationpad2d 。

    通过复制输入张量边界元素对输入张量进行填充操作。

    参数：
        - **padding** (Union[int, tuple, list]): 填充范围的大小。若输入是一个 int，那各个边界上都会填充上同样大小的数据。若输入是一个四个元素的元组，那么使用 (:math:`\\mathrm{padding_{left}}`, :math:`\\mathrm{padding_{right}}`, :math:`\\mathrm{padding_{top}}`, :math:`\\mathrm{padding_{bottom}}`)。

    形状：
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` ，其中

            :math:`H_{out} = H_{in} + \\mathrm{padding_{top}} + \\mathrm{padding_{bottom}}`

            :math:`W_{out} = W_{in} + \\mathrm{padding_{left}} + \\mathrm{padding_{right}}`

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> m = flow.nn.ReplicationPad2d((2, 2, 1, 1))
        >>> input = flow.tensor(np.arange(18).reshape((1, 2, 3, 3)).astype(np.float32))
        >>> input_int = flow.tensor(np.arange(18).reshape((1, 2, 3, 3)).astype(np.int32))
        >>> output = m(input)
        >>> output.shape
        oneflow.Size([1, 2, 5, 7])
        >>> output
        tensor([[[[ 0.,  0.,  0.,  1.,  2.,  2.,  2.],
                  [ 0.,  0.,  0.,  1.,  2.,  2.,  2.],
                  [ 3.,  3.,  3.,  4.,  5.,  5.,  5.],
                  [ 6.,  6.,  6.,  7.,  8.,  8.,  8.],
                  [ 6.,  6.,  6.,  7.,  8.,  8.,  8.]],
        <BLANKLINE>
                 [[ 9.,  9.,  9., 10., 11., 11., 11.],
                  [ 9.,  9.,  9., 10., 11., 11., 11.],
                  [12., 12., 12., 13., 14., 14., 14.],
                  [15., 15., 15., 16., 17., 17., 17.],
                  [15., 15., 15., 16., 17., 17., 17.]]]], dtype=oneflow.float32)

    """
)

## oneflow.nn.SELU

reset_docstr(
    oneflow.nn.Sequential,
    r"""Sequential(*args: Any)
    
    一个序列容器。

    按照 Module 在构造函数中被传递的顺序将其添加到容器中。或者，也可以向构造函数传递 Module 的有序字典。

    为了便于理解，这里有一个示例：

    .. code-block:: python

        >>> import oneflow.nn as nn
        >>> from collections import OrderedDict
        >>> nn.Sequential(nn.Conv2d(1,20,5), nn.ReLU(), nn.Conv2d(20,64,5), nn.ReLU()) #doctest: +ELLIPSIS
        Sequential(
          (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
          (1): ReLU()
          (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
          (3): ReLU()
        )
        >>> nn.Sequential(OrderedDict([
        ...    ('conv1', nn.Conv2d(1,20,5)),
        ...    ('relu1', nn.ReLU()),
        ...    ('conv2', nn.Conv2d(20,64,5)),
        ...    ('relu2', nn.ReLU())
        ... ])) #doctest: +ELLIPSIS
        Sequential(
          (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
          (relu1): ReLU()
          (conv2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
          (relu2): ReLU()
        )

    """
)

## oneflow.nn.SiLU
## oneflow.nn.Sigmoid

reset_docstr(
    oneflow.nn.SmoothL1Loss,
    """SmoothL1Loss(reduction: str = 'mean', beta: float = 1.0)
    
    若逐元素的绝对误差低于 beta ，则创建一个使用平方项的标准，否则创建一个使用 L1 项的标准。

    此接口与 PyTorch 一致。文档参考自： https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html 。
    
    与 :class:`torch.nn.MSELoss` 相比，它对异常值不太敏感，并在某些场景下可以防止梯度爆炸。比如 Ross Girshick 的论文`Fast R-CNN <https://openaccess.thecvf.com/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf>`__。

    对于大小为 :math:`N` 的批次，未减少的损失可以描述为：

    .. math::
        \\ell(x, y) = L = \\{l_1, ..., l_N\\}^T

    其中：

    .. math::
        l_n = \\begin{cases}
        0.5 (x_n - y_n)^2 / beta, & \\text{if } |x_n - y_n| < beta \\\\
        |x_n - y_n| - 0.5 * beta, & \\text{otherwise }
        \\end{cases}

    若 `reduction` == `none` ，那么：

    .. math::
        \\ell(x, y) =
        \\begin{cases}
            \\operatorname{mean}(L), &  \\text{if reduction} = \\text{`mean';}\\\\
            \\operatorname{sum}(L),  &  \\text{if reduction} = \\text{`sum'.}
        \\end{cases}

    .. note::
        平滑 L1 损失可以看作是 :class:`L1Loss` ，但 :math:`|x - y| < beta` 部分被替
            换为二次函数，使得它的斜率在 :math:`|x - y| = beta` 处为 1。二次函数的部
            分平滑了 :math:`|x - y| = 0` 处的 L1 损失。

    .. note::
        平滑 L1 损失与 :class:`HuberLoss` 密切相关，相当
            于 :math:`huber(x, y) / beta` （注意 Smooth L1 的 beta 超
            参数相当于 Huber 的 delta）。这导致了以下差异：

        * 当 beta 趋向 0，平滑 L1 损失收敛到 :class:`L1Loss` ，而 :class:`HuberLoss` 收敛到常数 0 损失。
        * 当 beta 趋向 :math:`+\\infty` ，平滑 L1 损失收敛到常数 0 损失，而 :class:`HuberLoss` 收敛到 :class:`MSELoss`。
        * 对于平滑 L1 损失，随着 beta 的变化，损失的 L1 段的斜率恒为 1。而对于 :class:`HuberLoss` ，斜率是 beta。

    参数：
        - **size_average** (bool, optional): 已弃用（参考 :attr:`reduction`）。默认情况下，损失是批次中每个损失元素的平均值。请注意，对于某些损失，每个样本有多个元素。若 :attr:`size_average` == ``False``，则每个小批量的损失相加。当 :attr:`reduce` == ``False`` 时忽略。默认值为 ``True``。
        - **reduce** (bool, optional): 已弃用（参考 :attr:`reduction`）。根据 :attr:`size_average` 对每个小批量的损失进行平均或汇总。若 :attr:`reduce` == ``False``，则返回每个批元素的损失，并忽略 :attr:`size_average`。默认值为 ``True``。
        - **reduction** (string, optional): 指定应用于输出的 reduction：``'none'`` | ``'mean'`` | ``'sum'``. ``'none'`` ：不进行 reduction；``'mean'`` ：输出的和将会除以输出中的元素数量；``'sum'`` ：输出将被求和。注意： :attr:`size_average` 和 :attr:`reduce` 正逐渐被弃用，指定这二者的任何一个都将覆盖 :attr:`reduction`。默认值为 ``'mean'``。
        - **beta** (float, optional): 指定在 L1 和 L2 损失之间更改的阈值。该值必须为非负。默认值为 1.0。

    形状：
        - Input: :math:`(N, *)` ，其中 `*` 表示任意数量的附加维度。
        - Target: :math:`(N, *)` ，与输入的形状相同。
        - Output: 若 :attr:`reduction` == ``'none'`` 则输出为形状为 :math:`(N)` 的张量，否则是一个标量。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> x = flow.tensor(np.array([0.1, 0.4, 0.3, 0.5, 0.9]).astype(np.float32), dtype=flow.float32)
        >>> y = flow.tensor(np.array([0.3, 0.9, 2.5, 0.4, 0.3]).astype(np.float32), dtype=flow.float32)
        >>> m = flow.nn.SmoothL1Loss(reduction="none")
        >>> out = m(x, y)
        >>> out
        tensor([0.0200, 0.1250, 1.7000, 0.0050, 0.1800], dtype=oneflow.float32)

        >>> m = flow.nn.SmoothL1Loss(reduction="mean")
        >>> out = m(x, y)
        >>> out
        tensor(0.4060, dtype=oneflow.float32)

        >>> m = flow.nn.SmoothL1Loss(reduction="sum")
        >>> out = m(x, y)
        >>> out
        tensor(2.0300, dtype=oneflow.float32)
    """
)

## oneflow.nn.Softmax
## oneflow.nn.Softplus
## oneflow.nn.Softsign

reset_docstr(
    oneflow.nn.Tanh,
    """

    此算子计算张量的双曲正切值。

    等式为：

    .. math::

        out = \\frac{e^x-e^{-x}}{e^x+e^{-x}}

    参数：
        - **input** (oneflow.Tensor): 张量。

    返回值：
        oneflow.Tensor: 运算结果。

    示例：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = np.array([-1, 0, 1]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> tanh = flow.nn.Tanh()
        >>> out = tanh(input)
        >>> out
        tensor([-0.7616,  0.0000,  0.7616], dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.nn.TripletMarginLoss,
    r"""TripletMarginLoss(margin: float = 1.0, p: float = 2.0, eps: float = 1e-06, swap: bool = False, size_average=None, reduce=None, reduction: str = 'mean')
    
    在给定输入张量 :math:`x1`, :math:`x2`, :math:`x3` 和值大于 :math:`0` 的边距的情况下，创建一个测量三元组损失的标准。这用于测量样本之间的相对相似性。 三元组由 `a`, `p` and `n` 组成（即分别为锚点、正例和负例）。所有输入张量的形状应为 :math:`(N, D)`

    在 V. Balntas, E. Riba 等人的 `Learning local feature descriptors with triplets and shallow convolutional neural networks <http://www.bmva.org/bmvc/2016/papers/paper119/index.html>`__ 中详细描述了距离交换。

    小批量中每一个样本的损失函数是：

    .. math::
        L(a, p, n) = \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\}


    其中

    .. math::
        d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p

    参数：
        - **margin** (float, optional): 默认值为 :math:`1`
        - **p** (float, optional): 成对距离的范数，默认值为 :math:`2.0`
        - **swap** (bool, optional): 默认值为 ``False``.
        - **reduction** (string, optional): 指定对输出应用的 reduction ：
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'`` ：不进行 reduction ；
            ``'mean'`` ：输出的和将会除以输出中的元素数量；
            ``'sum'`` ：输出将被求和。默认值为 ``'mean'``。注意： :attr:`size_average` 和 :attr:`reduce` 正逐渐被弃用，指定这二者的任何一个都将覆盖 :attr:`reduction`。默认值为 ``'mean'``

    形状：
        - Input: :math:`(N, D)` ，其中 :math:`D` 是向量维度。
        - Output: 若 :attr:`reduction` == ``'none'`` 则输出为形状为 :math:`(N)` 的张量，否则是一个标量。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> triplet_loss = flow.nn.TripletMarginLoss(margin=1.0, p=2)
        >>> anchor = np.array([[1, -1, 1],[-1, 1, -1], [1, 1, 1]])
        >>> positive = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> negative = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
        >>> output = triplet_loss(flow.Tensor(anchor), flow.Tensor(positive), flow.Tensor(negative))
        >>> output
        tensor(6.2971, dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.nn.Upsample,
    """
    
    此接口与 PyTorch 一致。文档参考自： https://pytorch.org/docs/1.9.0/_modules/torch/nn/modules/upsampling.html#Upsample 。

    对给定的多通道 1D（时间）、2D（空间）或 3D（体积）数据进行上采样。

    假定输入数据的形式为 小批量 x 通道 x [可选深度] x [可选高度] x 宽度。 因此，对于空间输入，我们期待一个 4D 张量；对于体积输入，我们期待一个 5D 张量。

    可用于上采样的算法分别是 3D、4D 和 5D 输入张量的最近邻和线性、双线性、双三次和三线性算法。

    可以给出 :attr:`scale_factor` 或目标输出大小来计算输出大小。（你不能同时给出，因为它是模棱两可的）。

    参数：
        - **size** (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional): 输出空间大小。
        - **scale_factor** (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional): 空间大小的乘数。若是 tuple 则需要匹配输入大小。
        - **mode** (str, optional): 上采样算法： ``'nearest'``,``'linear'``, ``'bilinear'``, ``'bicubic'`` 和 ``'trilinear'``。默认值为： ``'nearest'``。
        - **align_corners** (bool, optional): 若设置为 ``True`` ，则若为 True，则输入和输出张量的角像素对齐，从而保留这些像素的值。这仅在模式为 ``'linear'``, ``'bilinear'``, 或 ``'trilinear'`` 时有效。默认值为 False。

    形状：
        - Input: :math:`(N, C, W_{in})`, :math:`(N, C, H_{in}, W_{in})` 或 :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, W_{out})`, :math:`(N, C, H_{out}, W_{out})`
          or :math:`(N, C, D_{out}, H_{out}, W_{out})` ，其中

    .. math::
        D_{out} = \\left\\lfloor D_{in} \\times \\text{scale_factor} \\right\\rfloor

    .. math::
        H_{out} = \\left\\lfloor H_{in} \\times \\text{scale_factor} \\right\\rfloor

    .. math::
        W_{out} = \\left\\lfloor W_{in} \\times \\text{scale_factor} \\right\\rfloor

    .. warning::
        若 ``align_corners = True`` ，线性插值模式（线性、双线性、双三次和三线性）不会按比例对齐输出和输入像素，因此输出值可能取决于输入大小。这是 0.3.1 版本之前这些模式的默认行为。 0.3.1 版本之后的默认值行为是 ``align_corners = False``。有关其如何影响输出的具体示例，请参见下文。

    .. note::
        若需要下采样或者一般性的调整大小，应该使用 :func:`~nn.functional.interpolate`。

    示例：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> input = flow.tensor(np.arange(1, 5).reshape((1, 1, 2, 2)), dtype=flow.float32)
        >>> input = input.to("cuda")
        >>> m = flow.nn.Upsample(scale_factor=2.0, mode="nearest")
        >>> output = m(input)
        >>> output #doctest: +ELLIPSIS
        tensor([[[[1., 1., 2., 2.],
                  ...
                  [3., 3., 4., 4.]]]], device='cuda:0', dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.nn.UpsamplingBilinear2d,
    """
    
    对由多个输入通道组成的输入信号应用 2D bilinear upsampling。
    
    若要指定比例，需要 :attr:`size` 或 :attr:`scale_factor` 作为它的构造函数参数。

    若给定 :attr:`size` ，则它也是图像 `(h, w)` 的大小。

    参数：
        - **size** (int or Tuple[int, int], optional): 输出空间大小。
        - **scale_factor** (float or Tuple[float, float], optional): 空间大小的乘数。

    .. warning::
        对这个类的维护已经停止，请使用 :func:`~nn.functional.interpolate`。它等同于 ``nn.functional.interpolate(..., mode='bilinear', align_corners=True)``

    形状：
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` ，其中

    .. math::
        H_{out} = \\left\\lfloor H_{in} \\times \\text{scale_factor} \\right\\rfloor

    .. math::
        W_{out} = \\left\\lfloor W_{in} \\times \\text{scale_factor} \\right\\rfloor

    示例：

    .. code-block:: python

        > import numpy as np
        > import oneflow as flow

        > input = flow.tensor(np.arange(1, 5).reshape((1, 1, 2, 2)), dtype=flow.float32)
        > input = input.to("cuda")
        > m = flow.nn.UpsamplingBilinear2d(scale_factor=2.0)
        > output = m(input)
        > output #doctest: +ELLIPSIS
        tensor([[[[1.0000, 1.3333, 1.6667, 2.0000],
                  ...
                  [3.0000, 3.3333, 3.6667, 4.0000]]]], device='cuda:0', dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.nn.UpsamplingNearest2d,
    """
    
    对由多个输入通道组成的输入信号应用 2D nearest neighbor upsampling。

    若要指定比例，需要 :attr:`size` 或 :attr:`scale_factor` 作为它的构造函数参数。

    若给定 :attr:`size` ，则它也是图像 `(h, w)` 的大小。

    参数：
        - **size** (int or Tuple[int, int], optional): 输出空间大小。
        - **scale_factor** (float or Tuple[float, float], optional): 空间大小的乘数。

    .. warning::
        对这个类的维护已经停止，请使用 :func:`~nn.functional.interpolate`。

    形状：
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` ，其中

    .. math::
          H_{out} = \\left\\lfloor H_{in} \\times \\text{scale_factor} \\right\\rfloor
          W_{out} = \\left\\lfloor W_{in} \\times \\text{scale_factor} \\right\\rfloor

    示例：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input = flow.tensor(np.arange(1, 5).reshape((1, 1, 2, 2)), dtype=flow.float32)
        >>> input = input.to("cuda")
        >>> m = flow.nn.UpsamplingNearest2d(scale_factor=2.0)
        >>> output = m(input)
        >>> output #doctest: +ELLIPSIS
        tensor([[[[1., 1., 2., 2.],
                  ...
                  [3., 3., 4., 4.]]]], device='cuda:0', dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.nn.ZeroPad2d,
    """
    
    此接口与 PyTorch 一致。文档参考自： https://pytorch.org/docs/stable/generated/torch.nn.ZeroPad2d.html 。

    用零填充输入张量边界。用户可以通过设置参数 `paddings` 来设置填充量。

    参数：
        - **padding** (Union[int, tuple]):  填充量的大小。若是 `int` 类型，则在所有边界中使用相同的填充。若是 4-`tuple` 则使用 (:math:`\\mathrm{padding_{left}}`, :math:`\\mathrm{padding_{right}}`, :math:`\\mathrm{padding_{top}}`, :math:`\\mathrm{padding_{bottom}}`)

    形状：
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` ，其中

            :math:`H_{out} = H_{in} + \\mathrm{padding_{top}} + \\mathrm{padding_{bottom}}`

            :math:`W_{out} = W_{in} + \\mathrm{padding_{left}} + \\mathrm{padding_{right}}`

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> m1 = flow.nn.ZeroPad2d(2)
        >>> m2 = flow.nn.ZeroPad2d((1,2,2,0))
        >>> input = flow.tensor(np.arange(18).reshape((1, 2, 3, 3)).astype(np.float32))
        >>> output = m1(input)
        >>> output.shape
        oneflow.Size([1, 2, 7, 7])
        >>> output
        tensor([[[[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  1.,  2.,  0.,  0.],
                  [ 0.,  0.,  3.,  4.,  5.,  0.,  0.],
                  [ 0.,  0.,  6.,  7.,  8.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0.,  0.,  0.]],
        <BLANKLINE>
                 [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  9., 10., 11.,  0.,  0.],
                  [ 0.,  0., 12., 13., 14.,  0.,  0.],
                  [ 0.,  0., 15., 16., 17.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0.,  0.,  0.]]]], dtype=oneflow.float32)
        >>> output = m2(input)
        >>> output
        tensor([[[[ 0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  1.,  2.,  0.,  0.],
                  [ 0.,  3.,  4.,  5.,  0.,  0.],
                  [ 0.,  6.,  7.,  8.,  0.,  0.]],
        <BLANKLINE>
                 [[ 0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  9., 10., 11.,  0.,  0.],
                  [ 0., 12., 13., 14.,  0.,  0.],
                  [ 0., 15., 16., 17.,  0.,  0.]]]], dtype=oneflow.float32)
    """
)

#### oneflow.nn.parallel.DistributedDataParallel
## oneflow.nn.utils.clip_grad_norm_
## oneflow.nn.utils.weight_norm
## oneflow.nn.utils.remove_weight_norm
