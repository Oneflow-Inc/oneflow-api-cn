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
