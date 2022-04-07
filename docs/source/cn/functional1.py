import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow._C.triplet_margin_loss,
    r"""
    文档参考自：https://pytorch.org/docs/stable/generated/torch.nn.functional.triplet_margin_loss.html?highlight=triplet_margin_loss

    在给定输入张量 :math:`x1`, :math:`x2`, :math:`x3` 和值大于 :math:`0` 的边距的情况下，创建一个测量三元组损失的标准。这用于测量样本之间的相对相似性。三元组由 `a`, `p` 和 `n` 组成（即分别为锚点、正例和负例）。所有输入张量的形状应为 :math:`(N, D)` 。

    Vassileios Balntas、Edgar Riba 等人的 `Learning shallow convolutional feature descriptors with triplet losses <http://www.bmva.org/bmvc/2016/papers/paper119/index.html>`__ 一文中详细描述了距离交换。

    小批量中每个样本的损失函数为：

    .. math::
        L(a, p, n) = \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\}


    其中

    .. math::
        d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p

    参数：
        - **margin** (float, optional) - 默认为 :math:`1`
        - **p** (float, optional) - 成对距离的范数，默认为 :math:`2.0`
        - **swap** (bool, optional) - 默认为 ``False`` 。V. Balntas、E. Riba 等人的 `Learning shallow convolutional feature descriptors with triplet losses <http://www.bmva.org/bmvc/2016/papers/paper119/index.html>`__ 一文中详细描述了距离交换。
        - **reduction** (string, optional) - 指定应用于输出的 reduction：``'none'`` | ``'mean'`` | ``'sum'``。若值为 ``'none'`` ：不进行 reduction；值为 ``'mean'`` ：输出的和将会除以输出中的元素数量；值为 ``'sum'`` ：输出将被求和。请注意：:attr:`size_average` 和 :attr:`reduce` 正逐渐被弃用，指定这二者的任何一个都将覆盖 :attr:`reduction`。默认值为 ``'mean'``。

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
    
    """,
)

reset_docstr(
    oneflow.nn.functional.dropout,
    """dropout(x: Tensor, p: float = 0.5, training: bool = True, generator :Generator = None, *, addend: Tensor) -> Tensor 
    
    文档引用自： https://pytorch.org/docs/stable/generated/torch.nn.functional.dropout.html 。

    在训练期间，使用来自伯努利分布的样本以概率 :attr:`p` 将输入张量的一些元素随机归零。

    参数：      
        - **x** (Tensor) - 将应用 dropout 的张量。
        - **p** (float) - 任一元素被归零的概率，默认为 0.5。
        - **training** (bool) - 若为 True 则应用 dropout，默认为 True。   
        - **generator** (Generator, optional) - 用于采样的伪随机数发生器。
        - **addend** (Tensor, optional) - 加入到 dropout 结果中的张量，它可以用于模型的残差连接结构。默认为 None。

    形状：
        - Input: :math:`(*)` ，输入可以为任何形状。
        - Output: :math:`(*)` ，与输入形状相同。

    示例 1：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

       
        >>> arr = np.array(
        ...    [
        ...        [-0.7797, 0.2264, 0.2458, 0.4163],
        ...        [0.4299, 0.3626, -0.4892, 0.4141],
        ...        [-1.4115, 1.2183, -0.5503, 0.6520],
        ...    ]
        ... )
        >>> x = flow.tensor(arr, dtype=flow.float32)
        >>> y = flow.nn.functional.dropout(x, p=0) 

        >>> arr = np.array(
        ...    [
        ...        [-0.7797, 0.2264, 0.2458, 0.4163],
        ...        [0.4299, 0.3626, -0.4892, 0.4141],
        ...        [-1.4115, 1.2183, -0.5503, 0.6520],
        ...    ]
        ... )
        >>> x = flow.tensor(arr, dtype=flow.float32)
        >>> generator = flow.Generator()
        >>> y = flow.nn.functional.dropout(x, p=0.5, generator=generator) 
      
    示例 2：
    
    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

       
        >>> arr = np.array(
        ...    [
        ...        [-0.7797, 0.2264, 0.2458, 0.4163],
        ...        [0.4299, 0.3626, -0.4892, 0.4141],
        ...        [-1.4115, 1.2183, -0.5503, 0.6520],
        ...    ]
        ... )
        >>> x = flow.tensor(arr, dtype=flow.float32)
        >>> addend = flow.ones((3, 4), dtype=flow.float32)
        >>> y = flow.nn.functional.dropout(x, p=0, addend=addend) 
        >>> y #doctest: +ELLIPSIS
        tensor([[ 0.2203,  1.2264,  1.2458,  1.4163],
                [ 1.4299,  1.3626,  0.5108,  1.4141],
                [-0.4115,  2.2183,  0.4497,  1.6520]], dtype=oneflow.float32)
    
    参考 :class:`~oneflow.nn.Dropout` 获得更多细节。
 
    """,
)

reset_docstr(
    oneflow._C.upsample,
    r"""upsample(x: Tensor, height_scale: Float, width_scale: Float, align_corners: Bool, interpolation: str, data_format: str = "channels_first") -> Tensor
  
    对给定的多通道 2D（空间）数据进行上采样。

    假设输入数据的形式为 `minibatch x channels x height x width`。因此，对于空间输入，4D 张量是被期待的。

    可用于上采样的算法分别是最近邻、双线性、4D 输入张量。

    参数：
        - **height_scale** (float) - 空间大小的乘数。如果它是元组，则必须匹配输入大小。
        - **width_scale** (float) - 空间大小的乘数。如果它是元组，则必须匹配输入大小。
        - **align_corners** (bool) - 如果为 ``True``，则输入和输出张量的角像素对齐，从而保留这些像素的值。这仅在模式为“双线性”时有效。  
        - **interpolation** (str, optional) - 上采样算法，可以为 ``'nearest'`` 或 ``'bilinear'``。
        - **data_format** (str, optional) - 默认为 ``'channels_first'``。

    形状：
        - Input: : :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` ，其中
  
    .. math::
        H_{out} = \left\lfloor H_{in} \times \text{height_scale} \right\rfloor

    .. math::
        W_{out} = \left\lfloor W_{in} \times \text{width_scale} \right\rfloor

  
    示例：

    .. code-block:: python

        >>> import numpy as np

        >>> import oneflow as flow

        >>> input = flow.tensor(np.arange(1, 5).reshape((1, 1, 2, 2)), dtype=flow.float32)  
        >>> output = flow.nn.functional.upsample(input, height_scale=2.0, width_scale=2.0, align_corners=False, interpolation="nearest")
    
        >>> output
        tensor([[[[1., 1., 2., 2.],
                  [1., 1., 2., 2.],
                  [3., 3., 4., 4.],
                  [3., 3., 4., 4.]]]], dtype=oneflow.float32)

    参考 :class:`~oneflow.nn.Upsample` 获得更多细节。

    """,
)

reset_docstr(
    oneflow.nn.functional.affine_grid,
    """此接口与 PyTorch 一致。文档参考自： https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html?highlight=affine_grid#torch.nn.functional.affine_grid 。

    给定一批仿射矩阵 :attr:`theta` ，生成 2D 或 3D 流场（采样网格）。

    .. note::
        此函数通常与 :func:`grid_sample` 结合使用来构建 `Spatial Transformer Networks`_ 。

    参数：
        - **theta** (Tensor) - 形状为 (:math:`N, 2, 3`)（2D 场景）或形状为 (:math:`N, 3, 4`)（3D 场景）的仿射矩阵的输入批量。
        - **size** (oneflow.Size) - 目标输出图像大小。形状为 (:math:`N, C, H, W`)（2D 场景）或 (:math:`N, C, D, H, W`)（3D 场景），例如 ``oneflow.Size((32, 3, 24, 24))``。
        - **align_corners** (bool) - 如果为 ``True``，则考虑 ``-1``和 ``1`` 来指代角像素的中心，而不是图像的角。参考 :func:`grid_sample` 来获得更详细的描述。由 :func:`affine_grid` 生成的网格应使用与此选项相同的设置传递给 :func:`grid_sample`。默认为 ``False``。

    返回值：
        output (Tensor) - 形状为 (:math:`N, H, W, 2`) 的输出张量。

    .. _`Spatial Transformer Networks`:
        https://arxiv.org/abs/1506.02025

    Examples::

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.tensor(np.arange(1., 7).reshape((1, 2, 3)), dtype=flow.float32)
        >>> output = flow.nn.functional.affine_grid(input, flow.Size([1, 1, 2, 2]), align_corners=True)
        >>> output
        tensor([[[[ 0., -3.],
                  [ 2.,  5.]],
        <BLANKLINE>
                 [[ 4.,  7.],
                  [ 6., 15.]]]], dtype=oneflow.float32)
    """
)

reset_docstr(
    oneflow.nn.functional.grid_sample,
    """此接口与 PyTorch 一致。文档参考自： https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html?highlight=grid_sample#torch.nn.functional.grid_sample

    给定 :attr:`input` 和流场 :attr:`grid`，使用 :attr:`input` 和 :attr:`grid` 中的像素位置计算输出。目前，仅支持空间 (4-D) 和体积 (5-D) 输入。在空间 (4-D) 情况下，对于形状为 :math:`(N, C, H_{in}, W_{in})` 的输入和形状为 :math:`(N, H_{out}, W_{out}, 2)` 的网格，输出将具有形状 :math:`(N, C, H_{out}, W_{out})`。

    对于每个输出位置 ``output[n, :, h, w]``，大小为 2 的向量 ``grid[n, h, w]`` 指定输入像素位置 ``x`` 和 ``y``，用于插值输出值 ``output[n, :, h, w]``。在 5D 输入的情况下，``grid[n, d, h, w]`` 指定用于插值 ``output[n, :, d, h, w]`` 的 ``x``、``y``、``z`` 像素位置。:attr:`mode` 参数指定对输入像素进行采样的最近或双线性插值方法。网格指定由输入空间维度归一化的采样像素位置。因此，它应该具有 ``[-1, 1]`` 范围内的大多数值。例如，``x = -1, y = -1`` 是输入的左上角像素，``x = 1, y = 1`` 是输入的右下角像素。

    如果 :attr:`grid` 的值在 ``[-1, 1]`` 之外，相应的输出按照 :attr:`padding_mode` 的定义进行处理。选项是：
        - ``padding_mode="zeros"``: 使用 ``0`` 表示越界网格位置。
        - ``padding_mode="border"``: 使用边界值表示越界网格位置。
        - ``padding_mode="reflection"``: 使用边界反映的位置处的值表示越界网格位置。对于远离边界的位置，它将一直被反射直到进入边界。例如（归一化）像素位置 ``x = -3.5`` 由边界 ``-1`` 反射并变为 ``x = 1.5``，然后由边界 ``1`` 反射并变为 ``x = -0.5``。

    Note:
        此函数通常与 :func:`grid_sample` 结合使用来构建 `Spatial Transformer Networks`_ 。

    Note:
        :attr:`grid` 中的 NaN 值将被解释为 ``-1``。

    参数：
        - **input** (Tensor) - 形状为 :math:`(N, C, H_{in}, W_{in})` （4D 情况下）或形状为 :math:`(N, C, D_{in}, H_{in}, W_{in})` （5D 情况下）的输入张量
        - **grid** (Tensor) - 形状为 :math:`(N, H_{out}, W_{out}, 2)` （4D 情况下）或形状为 :math:`(N, D_{out}, H_{out}, W_{out}, 3)` （5D 情况下）的流场
        - **mode** (str) - 计算输出值的插值模式：``'bilinear'`` | ``'nearest'`` | ``'bicubic'``。默认为 ``'bilinear'``。注意 ``mode='bicubic'`` 仅支持 4D 输入。当 ``mode='bilinear'`` 且输入是 5D 时，内部使用的插值模式实际上是三线性的。然而，当输入是 4D 时，插值模式将是双线性的。
        - **padding_mode** (str) - 外部网格值的填充模式：``'zeros'`` | ``'border'`` | ``'reflection'``。默认为 ``'zeros'``。
        - **align_corners** (bool) - 在几何上，我们将输入的像素视为正方形而不是点。如果设置为 ``True``，则极值 （``-1``  和 ``1``）被认为指代输入角像素的中心点。如果设置为 ``False``，则它们被认为是指输入角像素的角点，从而使采样与分辨率无关。 此选项与 :func:`interpolate` 中的 ``align_corners`` 选项一致，因此此处使用的任何选项也应用于在网格采样之前调整输入图像的大小。默认为 ``False``。

    返回值：
        output (Tensor) - 输出张量。

    .. _`Spatial Transformer Networks`:
        https://arxiv.org/abs/1506.02025

    .. note::
        ``mode='bicubic'`` 使用 :math:`\\alpha=-0.75` 的 `三次卷积算法`_ 实现。常数 :math:`\\alpha` 可能因包而异。例如 `PIL`_ 和 `OpenCV`_ 分别使用 -0.5 和 -0.75。该算法可能会``“超出”``它的插值范围。例如，输入位于 ``[0, 255]`` 中的插值时，它可能会产生负值或大于 `255` 的值。使用 :func:`flow.clamp` 钳制结果以确保它们在有效范围内。
    .. _`三次卷积算法`: https://en.wikipedia.org/wiki/Bicubic_interpolation
    .. _`PIL`: https://github.com/python-pillow/Pillow/blob/4634eafe3c695a014267eefdce830b4a825beed7/src/libImaging/Resample.c#L51
    .. _`OpenCV`: https://github.com/opencv/opencv/blob/f345ed564a06178670750bad59526cfa4033be55/modules/imgproc/src/resize.cpp#L908
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.tensor(np.arange(1., 11).reshape((1, 1, 2, 5)), dtype=flow.float32)
        >>> np_grid = np.array(
        ...     [[[-0.9, -4.1], [0, 0.2000], [1, -1], [-0.333, 1e-6], [0.5, 1.0]],
        ...      [[-1.0, -0.5], [0, 0.3333], [1, -1], [-0.200, 1e-6], [1.5, 0.5]]]
        ... ).reshape(1, 2, 5, 2)
        >>> grid = flow.tensor(np_grid, dtype=flow.float32)
        >>> output = flow.nn.functional.grid_sample(input, grid, mode='nearest', padding_mode='zeros',
        ...                                        align_corners=True)
        >>> output
        tensor([[[[0., 8., 5., 7., 9.],
                  [1., 8., 5., 8., 0.]]]], dtype=oneflow.float32)
    """

)

reset_docstr(
    oneflow.nn.functional.interpolate,
    """
    此接口与 PyTorch 一致。文档参考自：https://pytorch.org/docs/1.9.0/_modules/torch/nn/functional.html#interpolate 。
    
    上/下采样输入到给定大小或给定 :attr:`scale_factor`。用于插值的算法由 :attr:`mode` 确定。
    目前支持时间、空间和体积采样，即预期输入为三维、四维或五维形状。

    输入大小解释为：``mini-batch x channels x [optional depth] x [optional height] x width``。

    可用于调整大小的模式有：最近、线性（仅限 3D）、双线性、双三次（仅限 4D）、三线性（仅限 5D）、面积。

    参数：
        - **input** (Tensor) - 输入张量。
        - **size** - 输出空间大小，可以为一个、两个或三个 int 组成的 ``tuple``。
        - **scale_factor** (float or Tuple[float]) - 空间大小的乘数。如果是元组，则必须匹配输入大小。
        - **mode** (str) - 选择用于上采样的算法：``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` | ``'trilinear'`` | ``'area'``。默认为 ``'nearest'``
        - **align_corners** (bool, optional) - 几何上，我们将输入和输出的像素视为正方形而不是点。如果设置为 ``True``，则输入和输出张量由其角像素的中心点对齐，保留角像素处的值。如果设置为 ``False``，则输入和输出张量通过其角像素的角点对齐，并且插值对边界外的值使用边缘值填充，当 :attr:`scale_factor` 保持相同时，此操作与输入大小无关。这仅在模式为“线性”、“双线性”、“双三次”或“三线性”时有效。默认值：``False``。
        - **recompute_scale_factor** (bool, optional) - 重新计算 ``scale_factor`` 以用于插值计算。当 ``scale_factor`` 作为参数传递时，它用于计算 ``output_size``。如果 ``recompute_scale_factor`` 为 ``False`` 或未指定，传入的 ``scale_factor`` 将用于插值计算。否则，将根据用于插值计算的输出和输入大小计算新的 ``scale_factor`` （即计算将与显式传入计算的 ``output_size`` 相同）。请注意，当 ``scale_factor`` 为浮点数时，由于舍入和精度问题，重新计算的 ``scale_factor`` 可能与传入的值不同。

    .. note::
        使用 ``mode='bicubic'``，可能会导致过冲，也就是它可以为图像产生负值或大于 255 的值。如果要减少显示图像时的过冲，请显式调用 ``result.clamp(min=0, max=255)`` 

    .. warning::
        若 ``align_corners = True``，线性插值模式（线性、双线性和三线性）不会按比例对齐输出和输入像素，因此输出值可能取决于输入大小。这是 0.3.1 之前这些模式的默认行为，此后的默认行为是 ``align_corners = False``。有关这如何影响输出的具体示例，请参见 :class:`~torch.nn.Upsample`

    .. warning::
        当指定 ``scale_factor`` 时，如果 ``recompute_scale_factor=True``，则 ``scale_factor`` 用于计算输出大小，然后用于推断插值的新比例。

    示例：

    .. code-block:: python

        > import oneflow as flow
        > import numpy as np

        > input = flow.tensor(np.arange(1, 5).reshape((1, 1, 4)), dtype=flow.float32)
        > output = flow.nn.functional.interpolate(input, scale_factor=2.0, mode="linear")
        > output
        tensor([[[1.0000, 1.2500, 1.7500, 2.2500, 2.7500, 3.2500, 3.7500, 4.0000]]], dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow._C.ctc_greedy_decoder,
    """对输入中给出的 logits 执行贪婪解码（最佳路径）。

    参数：
        - **log_probs** (oneflow.Tensor) - 形状为 ``[input_length, batch_size, num_labels]`` 的张量。输出的对数概率（例如，使用 ``flow.nn.logsoftmax()`` 获得）。
        - **input_lengths** (oneflow.Tensor) - 形状为 ``[batch_size]`` 的张量。它表示输入的长度。并且在序列被填充到相等长度的假设下，为每个序列指定长度以实现掩码。
        - **merge_repeated** (bool, optional) - 如果 ``merge_repeated`` 为 ``True``，则合并输出中的重复类。这意味着如果连续 logits 的最大索引相同，则仅发出其中的第一个。默认为 ``True``。

    返回值：
        - decoded(oneflow.Tensor) - 形状为 [batch_size, input_length] 的张量，解码后的输出。
        - neg_sum_logits(oneflow.Tensor) - 一个浮点矩阵 (batch_size x 1)，对于找到的序列，包含每个时间帧最大 logit 总和的负数。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> log_probs = flow.tensor(
        ...     [
        ...         [[-1.54, -1.20, -1.95, -1.65, -1.81], [-1.84, -1.74, -1.58, -1.55, -1.12]],
        ...         [[-1.68, -1.48, -1.89, -1.30, -2.07], [-1.13, -1.45, -1.24, -1.61, -1.66]],
        ...         [[-1.56, -1.40, -2.83, -1.67, -1.48], [-1.20, -2.01, -2.05, -1.95, -1.24]],
        ...         [[-2.09, -1.76, -1.36, -1.67, -1.45], [-1.85, -1.48, -1.34, -2.16, -1.55]],
        ...     ]
        ... )
        >>> input_lengths = flow.tensor([4, 4])
        >>> decoded, neg_sum_logits = flow.nn.functional.ctc_greedy_decoder(log_probs, input_lengths)
        >>> decoded
        tensor([[1, 3, 1, 2],
                [0, 2, 0, 0]], dtype=oneflow.int64)
        >>> neg_sum_logits
        tensor([[5.2600],
                [4.7900]], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.nn.functional.sparse_softmax_cross_entropy,
    """此接口与 TensorFlow 一致。文档参考自：https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits

    计算 `logits` 和标签之间的稀疏 softmax 交叉熵。    

    测量类别互斥（每个条目恰好属于一个类别）的离散分类任务中的概率误差。例如，每张 CIFAR-10 图像都标有一个且只有一个标签：图像可以是狗或卡车，但不能同时是两者。一个常见的用例是具有形状 `[batch_size, num_classes]` 的 `logits` 和具有形状 `[batch_size]` 的标签，但支持更高的维度，在这种情况下，假设第 `dim` 维度的大小为 `num_classes`。`logits` 的数据类型必须为 `float16`、`float32` 或 `float64`，标签的数据类型必须为 `int32` 或 `int64`。

    参数：
        - **labels** (Tensor) - 具有 ``[d_0, d_1, ..., d_{r-1}]`` 的形状（其中 `r` 是标签和输出的 rank），其数据类型是 `int32` 或 `int64`。标签中的每个条目必须是 ``[0, num_classes)`` 中的索引。
        - **logits** (Tensor) - 具有 ``[d_0, d_1, ..., d_{r-1}, num_classes]`` 的形状且数据类型是 `float16`、`float32` 或 `float64` 的每个标签激活（通常是线性输出）。这些激活能被解释为非标准化的对数概率。

    返回值：
        output (Tensor) - 与标签具有相同形状且与 `logits` 相同类型的张量，具有 softmax 交叉熵损失。

    示例：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> np_logits = np.array(
        ...      [
        ...          [2.0, -5.0, 0.5, -0.1],
        ...          [0.0, 0.0, 1.9, 1.4],
        ...          [-100.0, 100.0, -100.0, -100.0],
        ...      ]
        ...  )
        >>> np_labels = np.array([0, 3, 1])
        >>> logits = flow.tensor(np_logits, dtype=flow.float32)
        >>> labels = flow.tensor(np_labels, dtype=flow.int32)
        >>> output = flow.nn.functional.sparse_softmax_cross_entropy(
        ...     labels=labels, logits=logits
        ... )
        >>> output
        tensor([0.2975, 1.1448, -0.0000], dtype=oneflow.float32)
    """
)

reset_docstr(
    oneflow.nn.functional.embedding,
    r"""一个简单的查找表，可以在固定的字典和大小中查找嵌入项。
    该模块通常用于使用索引检索词嵌入。模块的输入是索引列表和嵌入矩阵，输出是相应的词嵌入。查看 :class:`oneflow.nn.Embedding` 获得更多细节。

    参数：
        - **input** (LongTensor) - 包含嵌入矩阵中的索引的张量。
        - **weight** (Tensor) - 行数等于最大可能索引 ``+1`` 且列数等于嵌入大小的嵌入矩阵。
        - **padding_idx** (int, optional) - 如果指定，则 :attr:`padding_idx` 处的条目不会影响梯度；因此 :attr:`padding_idx` 处的嵌入向量在训练期间不会更新，即它仍然是一个固定的 ``pad``。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import oneflow.nn.functional as F

        >>> # a batch of 2 samples of 4 indices each
        >>> input = flow.tensor([[1,2,4,5],[4,3,2,9]])
        >>> # an embedding matrix containing 10 tensors of size 3
        >>> embedding_matrix = flow.rand(10, 3)
        >>> output = F.embedding(input, embedding_matrix)
        >>> output.shape
        oneflow.Size([2, 4, 3])
        >>> # example with padding_idx
        >>> input = flow.tensor([[0,2,0,5]])
        >>> output = F.embedding(input, embedding_matrix, padding_idx=0)
        >>> output.shape
        oneflow.Size([1, 4, 3])
    """
)

reset_docstr(
    oneflow.nn.functional.linear,
    r"""对输入数据应用线性变换 :math:`y = xA^T + b`。

    形状：
        - Input - :math:`(N, *, in\_features)`，其中 `N` 表示批的大小，`*` 表示任意数量的附加维度。
        - Weight - :math:`(out\_features, in\_features)`
        - Bias - :math:`(out\_features)`
        - Output - :math:`(N, *, out\_features)`
    
    示例：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> input = flow.tensor(np.random.randn(128, 20))
        >>> weight = flow.tensor(np.random.randn(30, 20))
        >>> output = flow.nn.functional.linear(input, weight)
        >>> output.size()
        oneflow.Size([128, 30])
    
    """
)

reset_docstr(
    oneflow._C.cross_entropy,
    r"""
    文档参考自： https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html?highlight=nn%20functional%20cross_entropy#torch.nn.functional.cross_entropy 。
    查看 :class:`~oneflow.nn.CrossEntropyLoss` 获得更多细节。

    参数：
        input (Tensor) : :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
            in the case of K-dimensional loss. `input` is expected to contain unnormalized scores
            (often referred to as logits).
        target (Tensor) : If containing class indices, shape :math:`(N)` where each value is
            :math:`0 \leq \text{targets}[i] \leq C-1`, or :math:`(N, d_1, d_2, ..., d_K)` with
            :math:`K \geq 1` in the case of K-dimensional loss. If containing class probabilities,
            same shape as the input.
        weight (Tensor, optional) - a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        ignore_index (int, optional) - Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Note that
            :attr:`ignore_index` is only applicable when the target contains class indices.
            Default: -100
        reduction (string, optional) - Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import oneflow.nn.functional as F
        >>> input = flow.randn(3, 5, requires_grad=True)
        >>> target = flow.ones(3, dtype=flow.int64)
        >>> loss = F.cross_entropy(input, target)
        >>> loss.backward()

    """
)

reset_docstr(
    oneflow._C.log_softmax,
    r"""log_softmax(x: Tensor, dim: int) -> Tensor 

    LogSoftmax 的公式为：

    .. math::
        \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right) = x_i - \log({ \sum_j \exp(x_j)})
    
    参考 :class:`~oneflow.nn.LogSoftmax` 获得更多细节。
    """
)

reset_docstr(
    oneflow.nn.functional.gelu,
    r"""
    Gelu 激活算子，其公式为：

    .. math::
        out = 0.5 * x * (1 + tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715x^{3})))

    参数：
        **x** (oneflow.tensor) - 输入张量

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
    oneflow._C.glu,
    """
    glu(input: Tensor, dim: int) -> Tensor 

    glu 的公式为：

    .. math::
         GLU(input) = GLU(a, b) = a \otimes sigmoid(b)
    
    .. note::
        其中输入沿维度 dim 被切分成 a 和 b 两半，⊗ 是矩阵间的按元素积。
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import oneflow.nn as nn
        >>> x = flow.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=flow.float32)
        >>> y = nn.functional.glu(x)
        >>> y
        tensor([[0.9526, 1.9640],
                [4.9954, 5.9980]], dtype=oneflow.float32)

    参考 :class:`~oneflow.nn.GLU` 获得更多细节。
    """,
)

reset_docstr(
    oneflow._C.softsign,
    r"""
    softsign(x: Tensor) -> Tensor 

    softsign 的公式为：
    
    .. math::  
    
        softsign(x) = \frac{x}{1 + |x|}
    
    示例：
    
    .. code-block:: python
    
        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = np.array([1, 2, 3]).astype(np.float32)
        >>> input = flow.tensor(x) 
        >>> out = flow.nn.functional.softsign(input)
        >>> out
        tensor([0.5000, 0.6667, 0.7500], dtype=oneflow.float32)

    参考 :class:`~oneflow.nn.Softsign` 获得更多细节。
    
    """,
)

reset_docstr(
    oneflow._C.one_hot,
    r"""
    one_hot(input, num_classes=-1, on_value=1, off_value=0)
    该算子根据输入张量生成一个 onehot 张量。

    如果输入张量的秩为 `N`，相应的 onehot 张量的秩为 `N+1`。

    参数：
        - **input** (Tensor) - 输入张量。
        - **num_classes** (int) - 输出的 onehot 张量的长度。
        - **on_value** (Union[int, float], optional) - 当 `x[i] == i` 时的填充值，默认为 1。
        - **off_value** (Union[int, float], optional) - 当 `x[i] != i` 时的填充值，默认为 0。

    Note:
        输入张量的数据类型应为：`int32` 或 `int64`。

    返回值：
        oneflow.Tensor
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> input=flow.tensor(np.array([0, 3, 1, 2]).astype(np.int64), dtype=flow.int64)
        >>> out = flow.nn.functional.one_hot(input, num_classes=5)
        >>> out
        tensor([[1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0]], dtype=oneflow.int64)
    
    """
)
