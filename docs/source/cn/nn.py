import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.nn.CTCLoss,
    r"""CTCLoss(blank=0, reduction='mean', zero_infinity=False)

    计算 CTC(Connectionist Temporal Classification) 损失。
    此接口与 PyTorch 一致。可在 https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html#torch.nn.CTCLoss 
    找到相关文档。

    计算连续且未分段的时间序列和 :attr:`target` 序列之间的损失。CTCLoss 对 :attr:`input` 与 :attr:`target` 可能对齐的概率求和，产生一个损失值，该值相对于每个 :attr:`input` 节点是可导的。
    :attr:`input` 与 :attr:`target` 的对齐被假定为“多对一”，这限制了 :attr:`target` 序列的长度，即 :attr:`target` :math:`\leq` :attr:`input` 。
    
    参数：
        - **blank** (int, 可选的): 空白标签。 默认 :math:`0` 
        - **reduction** (string, 可选的): 指定应用于输出的简化，可以是 ``'none'`` 、 ``'mean'`` 或   ``'sum'`` ：
            ``'none'`` ：不进行简化； ``'mean'`` ：输出损失将除以目标长度，然后取批次的平均值。默认： ``'mean'`` 
        - **zero_infinity** (bool, 可选的): 是否设定无限损失和相关梯度归零。默认： ``False`` 。
            无限损失主要发生在 :attr:`input` 太短而无法与 :attr:`target` 对齐时

    形状：
        - **Log_probs** : 形状为 :math:`(T, N, C)` 的张量且 :math:`T = \text{input length}`  、 :math:`N = \text{batch size}` 、 :math:`C = \text{number of classes (including blank)}` 
        - **Targets** : 形状为 :math:`(N, S)` 或 :math:`(\operatorname{sum}(\text{target_lengths}))` 的张量，其中 :math:`N = \text{batch size}` 、 :math:`S = \text{max target length, if shape is } (N, S)` 。
          它代表 :attr:`target` 序列。 :attr:`target` 序列中的每个元素都是一个 class 索引。并且 :attr:`target` 索引不能为空（默认为 0）。在 :math:`(N, S)` 形式中，:attr:`target` 被填充到最长序列的长度并堆叠。
          在 :math:`(\operatorname{sum}(\text{target_lengths}))` 形式中，我们假设目标在 1 维内未填充和连接。
        - **Input_lengths** : 大小为 :math:`(N)` 的元组或张量，其中 :math:`N = \text{batch size}` 。它表示 :attr:`input` 的长度（每个都必须 :math:`\leq T` ）。假设序列被填充为相等长度的情况下，为每个序列指定长度以实现掩码。
        - **Target_lengths** : 大小为 :math:`(N)` 的元组或张量，其中 :math:`N = \text{batch size}` 。它代表 :attr:`target` 的长度。在假设序列被填充为相等长度的情况下，为每个序列指定长度以实现掩码。如果 :attr:`target` 形状是 :math:`(N,S)`，
          则 target_lengths 是每个目标序列的有效停止索引 :math:`s_n` ，这样每个目标序列都满足 ``target_n = targets[n,0:s_n]`` ，长度都必须 :math:`\leq S` 。
          如果目标是作为单个目标的串联的 1d 张量给出的，则 target_lengths 必须加起来为张量的总长度。

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

reset_docstr(
    oneflow.nn.CombinedMarginLoss,
    r"""CombinedMarginLoss(m1=1.0, m2=0.0, m3=0.0)

    以下操作在 InsightFace 中实现了 "margin_softmax" ：
    https://github.com/deepinsight/insightface/blob/master/recognition/arcface_mxnet/train.py
    InsightFace 中 margin_softmax 的实现是由多个算子组成的。
    我们将它们组合在一起以加快速度。

    参数：
        - **input** (oneflow.Tensor): 输入张量
        - **label** (oneflow.Tensor): 数据类型为整数的标签
        - **m1** (float): 损失参数 m1
        - **m2** (float): 损失参数 m2
        - **m3** (float): 损失参数 m3

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

    用常数值填充输入 tensor 边界。此接口与 PyTorch 一致，参考：https://pytorch.org/docs/stable/generated/torch.nn.ConstantPad1d.html?highlight=constantpad1d#torch.nn.ConstantPad1d

    用 :func:`torch.nn.functional.pad()` 来进行 `N` 维填充。

    参数:
        - **padding** (int, list, tuple): 填充的大小。如果数据类型为 `int` 则在两个边界中使用相同的填充。如果是 2-`tuple` ，则 (:math:`\text{padding_left}`, :math:`\text{padding_right}`) 
        - **value** (int, float): 用于填充的常量值。默认为 0

    形状：
        - **Input** : :math:`(N, C, W_{in})`
        - **Output** : :math:`(N, C, W_{out})` 其中

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
    
    此接口与 PyTorch 一致。文档可以参考：
    https://pytorch.org/docs/stable/generated/torch.nn.ZeroPad2d.html

    用 0 填充输入张量边界。用户可以通过设置参数 :attr:`paddings` 来设置填充量。

    参数：
        - **padding** (int 或 tuple): 填充的大小。如果是 `int`，则在所有边界中使用相同的填充。如果一个 2-`tuple` ，则(:math:`\mathrm{padding_{left}}`, :math:`\mathrm{padding_{right}}`, :math:`\mathrm{padding_{top}}`, :math:`\mathrm{padding_{bottom}}`)

    形状：
        - **Input** : :math:`(N, C, H_{in}, W_{in})`
        - **Output** : :math:`(N, C, H_{out}, W_{out})` 其中

            :math:`H_{out} = H_{in} + \mathrm{padding_{top}} + \mathrm{padding_{bottom}}`

            :math:`W_{out} = W_{in} + \mathrm{padding_{left}} + \mathrm{padding_{right}}`

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

    用常数值填充输入 tensor 边界。此接口与 PyTorch 一致，参考：https://pytorch.org/docs/stable/generated/torch.nn.ConstantPad1d.html?highlight=constantpad1d#torch.nn.ConstantPad1d

    用 :func:`torch.nn.functional.pad()` 来进行 `N` 维填充。

    参数:
        - **padding** (int, list, tuple): 填充的大小。如果数据类型为 `int` 则在所有边界中使用相同的填充。如果是 6-`tuple` ，
            则 (:math:`\text{padding_left}`, :math:`\text{padding_right}`,
            :math:`\text{padding_top}`, :math:`\text{padding_bottom}`,
            :math:`\text{padding_front}`, :math:`\text{padding_back}`)
        - **value** (int, float): 用于填充的常量值。默认为 0

    形状：
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` 其中

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
    
    此接口与 PyTorch 一致。
    文档参考自：https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html#conv1d
    
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

    * :attr:`dilation` 控制核心点 (kernel points) 之间的间距，也称为 `à trous algorithm` 。此操作很难描述，
      但是 `link`_ 很好的将 :attr:`dilation` 的作用可视化。

    Note:
        ``padding='valid'`` 等同于无填充。 ``padding='same'`` 填充输入，使输出具有与输入相同的形状。
        但是此种情况下不支持除了 1 以外的任何步幅 (stride) 值。

    参数：
        - **in_channels** (int): 输入图像的通道数
        - **out_channels** (int): 卷积产生的通道数
        - **kernel_size** (int 或者 tuple): 卷积核的大小
        - **stride** (int or tuple, 可选的): 卷积的步幅 (stride) 。默认： 1
        - **padding** (int, tuple 或者 str, 可选的): 添加到输入两侧的填充值。默认： 0
        - **padding_mode** (string, 可选的): 默认： ``'zeros'``
        - **dilation** (int or tuple, 可选的): 核心的元素之间的间距。默认： 1
        - **groups** (int, 可选的): 从输入通道到输出通道的 `blocked connections` 数。默认：1
        - **bias** (bool, 可选的): 如果为 ``True`` ，则向输出添加可学习的偏差。默认：``True``

    形状：
        - **Input** : :math:`(N, C_{in}, L_{in})`
        - **Output** : :math:`(N, C_{out}, L_{out})` 其中

          .. math::
              L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Attributes:
        weight (Tensor): 形状为 :math:`(\text{out\_channels}, 
            \frac{\text{in\_channels}}{\text{groups}}, \text{kernel\_size})` 的模块可学习权重。
            这些权重的值是由公式 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 计算而来，其中
            :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}` 

        bias (Tensor):  形状为 (out_channels) 的模块可学习权重。如果 :attr:`bias` 为 ``True`` ，
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
    
    此接口与 PyTorch 一致。
    文档参考自：https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html#conv2d
    
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

    * :attr:`dilation` 控制核心点 (kernel points) 之间的间距，也称为 `à trous algorithm` 。此操作很难描述，
      但是 `link`_ 很好的将 :attr:`dilation` 的作用可视化。

    * :attr:`groups` 控制输入和输出之间的连接。 :attr:`in_channels` 和 :attr:`out_channels` 都必须能被 :attr:`groups` 整除。
      例如，

        * 当 groups=1 时，所有输入都卷积到输出。

        * 当 groups=2 时，该操作等效于并排放置两个 conv 层，其中每个层检查一半的输入通道并产生一半的输出通道，然后将两者连接起来。

        * 当 groups= :attr:`in_channels` 时， 每个输入通道都与它自己的一组过滤器(大小为
          :math:`\frac{\text{out_channels}}{\text{in_channels}}`)进行卷积。

    参数 :attr:`kernel_size` 、 :attr:`stride` 、 :attr:`padding` 、 :attr:`dilation` 可以是：

        - 单个 ``int`` -- 在这种情况下，高度和宽度使用相同的值
        - 两个整数的``tuple`` -- 在这种情况下，第一个 `int` 用于高度，第二个 `int` 用于宽度

    Note:
        当 `groups == in_channels` 并且 `out_channels == K * in_channels` 时，其中 `K` 是一个正整数，这个操作被称为“深度卷积”。

        换句话说，对于大小为 :math:`(N, C_{in}, L_{in})` 的输入，可以使用参数 :math:`(C_\text{in}=C_\text{in}, C_\text{out}=C_\text{in} \times \text{K}, ..., \text{groups}=C_\text{in})` 
        执行具有深度乘数 `K` 的深度卷积。

    参数：
        - **in_channels** (int): 输入图像的通道数
        - **out_channels** (int): 卷积产生的通道数
        - **kernel_size** (int 或者 tuple): 卷积核的大小
        - **stride** (int or tuple, 可选的): 卷积的步幅 (stride) 。默认： 1
        - **padding** (int, tuple 或者 str, 可选的): 添加到输入两侧的填充值。默认： 0
        - **padding_mode** (string, 可选的): 默认： ``'zeros'``
        - **dilation** (int or tuple, 可选的): 核心的元素之间的间距。默认： 1
        - **groups** (int, 可选的): 从输入通道到输出通道的 `blocked connections` 数。默认：1
        - **bias** (bool, 可选的): 如果为 ``True`` ，则向输出添加可学习的偏差。默认：``True``

    形状：
        - **Input** : :math:`(N, C_{in}, H_{in}, W_{in})`
        - **Output** : :math:`(N, C_{out}, H_{out}, W_{out})` 其中

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

        bias (Tensor):  形状为 (out_channels) 的模块可学习权重。如果 :attr:`bias` 为 ``True`` ，
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

    此接口与 PyTorch 一致。
    文档参考自：https://pytorch.org/docs/master/generated/torch.nn.Conv3d.html#conv3d

    对由多个平面组成的输入信号应用 3D 卷积。

    在最简单的情况下，大小为 :math:`(N, C_{in}, D, H, W)` 的输入层的输出值和输出 
    :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` 可以被准确的表述为：

    .. math::
        out(N_i, C_{out_j}) = bias(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} weight(C_{out_j}, k) \star input(N_i, k)

    其中 :math:`\star` 为有效的 3D `cross-correlation`_ 运算符。

    * :attr:`stride` 是控制互相关 (cross-correlation) 的步幅 (stride) 的单个数字或单元素元组。

    * :attr:`padding` 控制应用于输入的填充量。可以是 `string` {{'valid', 'same'}}
      或一个给出在两侧的隐式填充量的整数元组。

    * :attr:`dilation` 控制核心点 (kernel points) 之间的间距，也称为 `à trous algorithm` 。此操作很难描述，
      但是 `link`_ 很好的将 :attr:`dilation` 的作用可视化。

    参数 :attr:`kernel_size` 、 :attr:`stride` 、 :attr:`padding` 、 :attr:`dilation` 可以是：

        - 单个 ``int`` -- 在这种情况下，长度、宽度和高度使用相同的值
        - 两个整数的``tuple`` -- 在这种情况下，第一个 `int` 用于长度，第二个 `int` 用于高度，第三个 `int` 用于宽度。

    Note:
        ``padding='valid'`` 等同于无填充。 ``padding='same'`` 填充输入，使输出具有与输入相同的形状。
        但是此种情况下不支持除了 1 以外的任何步幅 (stride) 值。

    参数：
        - **in_channels** (int): 输入图像的通道数
        - **out_channels** (int): 卷积产生的通道数
        - **kernel_size** (int 或者 tuple): 卷积核的大小
        - **stride** (int or tuple, 可选的): 卷积的步幅 (stride) 。默认： 1
        - **padding** (int, tuple 或者 str, 可选的): 添加到输入两侧的填充值。默认： 0
        - **padding_mode** (string, 可选的): 默认： ``'zeros'``
        - **dilation** (int or tuple, 可选的): 核心的元素之间的间距。默认： 1
        - **groups** (int, 可选的): 从输入通道到输出通道的 `blocked connections` 数。默认：1
        - **bias** (bool, 可选的): 如果为 ``True`` ，则向输出添加可学习的偏差。默认：``True``
    
    形状：
        - **Input** : :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - **Output** : :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` 其中

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

        bias (Tensor):  形状为 (out_channels) 的模块可学习权重。如果 :attr:`bias` 为 ``True`` ，
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
