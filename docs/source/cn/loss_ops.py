import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.repeat,
    r"""repeat(input, *sizes) -> Tensor
    
    沿指定维度通过重复使 :attr:`input` 尺寸变大，并返回。

    参数：
        - **x** (oneflow.tensor): 输入张量
        - ***size** (flow.Size 或 int): 沿每个维度重复的次数

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([[[[0, 1]],
        ...                       [[2, 3]],
        ...                       [[4, 5]]]], dtype=flow.int32)
        >>> out = input.repeat(1, 1, 2, 2)
        >>> out
        tensor([[[[0, 1, 0, 1],
                  [0, 1, 0, 1]],
        <BLANKLINE>
                 [[2, 3, 2, 3],
                  [2, 3, 2, 3]],
        <BLANKLINE>
                 [[4, 5, 4, 5],
                  [4, 5, 4, 5]]]], dtype=oneflow.int32)
    """
)

reset_docstr(
    oneflow.tile,
    r"""tile(input, reps) -> Tensor
    
    
    此接口与 PyTorch 一致。
    文档参考自：
    https://pytorch.org/docs/stable/generated/torch.tile.html


    
    通过重复 :attr:`input` 的元素构造一个新张量。 :attr:`reps` 参数指定每个维度的重复次数。

    如果 :attr:`reps` 的长度小于 :attr:`input` 的维度，则在 :attr:`reps` 前添加 1 。直到 :attr:`reps` 的长度
    等于 :attr:`input` 的维度。例如： :attr:`input` 的形状为  (8, 6, 4, 2)  ，而 :attr:`reps` 为 (2, 2) ，
    则 :attr:`reps` 被认为是 (1, 1, 2, 2) 。

    类似地，如果 :attr:`input` 的维度少于 :attr:`reps` 指定的维度，则 :attr:`input` 被视为在维度 0 处未压缩，
    直到它的维度与 :attr:`reps` 指定的一样多。例如，如果 :attr:`input` 的形状为 (4, 2) 而 ``reps`` 为 (3, 3, 2, 2)，
    则视 :attr:`input` 形状为 (1, 1, 4, 2)。

    .. note::
        这个函数类似于 NumPy 的 tile 函数。

    参数：
        - **input** (oneflow.tensor): 要重复元素的张量
        - **reps** (元组): 每个维度要重复的次数

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([1, 2], dtype=flow.int32)
        >>> out = input.tile(reps=(2,))
        >>> out
        tensor([1, 2, 1, 2], dtype=oneflow.int32)

        >>> input = flow.randn(5, 2, 1)
        >>> out = input.tile(reps=(3, 4))
        >>> out.size()
        oneflow.Size([5, 6, 4])
    """
)

reset_docstr(
    oneflow.nn.KLDivLoss,
    r"""KLDivLoss(reduction='mean', log_target=False)
    
    此接口与 PyTorch 一致。
    文档参考自： https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html?highlight=kldivloss#torch.nn.KLDivLoss

    测量 KL 散度。 

    `Kullback-Leibler divergence`_ 可用于连续分布中的距离测量，并且在对（离散采样）
    连续输出分布的空间执行直接回归时通常很有用。

    与 :class:`~torch.nn.NLLLoss` 一样， :attr:`input` 应包含 *log-probabilities* 并且不限于 2D tensor。

    
    默认情况下，目标被解释为 *probabilities* ，
    但可以将其视为将 :attr:`log_target` 设置为 ``True`` 的 *log-probabilities* 。

    此 criterion 要求 `target` 、 `Tensor` 的形状与 `input` 、 `Tensor` 一致。

    未简化 （即 :attr:`reduction` 设置为 ``'none'`` ） 的损失可以描述为：

    .. math::
        l(x,y) = L = \{ l_1,\dots,l_N \}, \quad
        l_n = y_n \cdot \left( \log y_n - x_n \right)

    其中索引 :math:`N` span ``input`` 的所有维度，并且 :math:`L` 具有与 ``input`` 相同的形状。
    如果 :attr:`reduction` 不为 ``'none'`` （默认 ``'mean'`` ），则：

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';} \\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}

    如果 :attr:`reduction` 为默认值 ``'mean'`` ，每个 minibatch 的损失在 observations 和维度上取平均值。
    如果 :attr:`reduction` 为 ``'batchmean'`` ，可以得到正确的 KL 散度，其中损失仅在批次维度上进行平均。
    ``'mean'`` 模式的行为将在下一个主要版本中更改为与 ``'batchmean'`` 相同。 

    .. _`kullback-leibler divergence`: https://en.wikipedia.org/wiki/Kullback-Leibler_divergence

    参数：
        - **reduction** (string, 可选的):  指定应用于输出的简化（可以为 ``'none'`` 、 ``'batchmean'`` 、 ``'sum'`` 、 ``'mean'`` ，默认： ``'mean'`` ）：
            - ``'none'`` ：不会进行简化
            - ``'batchmean'`` ：输出的总和将除以 batchsize 。
            - ``'sum'`` ：将输出求和。
            - ``'mean'`` ：输出和将除以输出中的元素数。
        - **log_target** (bool, 可选的): 指定是否在 log space 中传递 `target`。默认： ``False`` 

    .. note::
        :attr:`reduction` = ``'mean'`` 时不会返回真正的 KL 散度值，请使用符合 KL 数学定义的 :attr:`reduction` = ``'batchmean'`` 。
        在下一个主要版本中，``'mean'`` 将更改为与 ``'batchmean'`` 相同。
        
    形状：
        - **Input** : :math:`(N, *)` 其中 :math:`*` 表示任意数量的额外维度
        - **Target** : :math:`(N, *)`，与输入的形状相同
        - **Output** : 默认为标量。如果 :attr:``reduction`` 为 ``'none'`` ，则为 :math:`(N, *)`，形状与输入相同

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.tensor([-0.9021705, 0.08798598, 1.04686249], dtype=flow.float32)
        >>> target = flow.tensor([1.22386942, -0.89729659, 0.01615712], dtype=flow.float32)
        >>> m = flow.nn.KLDivLoss(reduction="none", log_target=False)
        >>> out = m(input, target)
        >>> out
        tensor([ 1.3514,  0.0000, -0.0836], dtype=oneflow.float32)
        >>> m = flow.nn.KLDivLoss(reduction="mean", log_target=False)
        >>> out = m(input, target)
        >>> out
        tensor(0.4226, dtype=oneflow.float32)
        >>> m = flow.nn.KLDivLoss(reduction="sum", log_target=True)
        >>> out = m(input, target)
        >>> out
        tensor(5.7801, dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.nn.L1Loss,
    r"""L1Loss(reduction='mean')
    
    此运算符计算 :attr:`input` 和 :attr:`target` 中每个元素之间的 L1 Loss 。

    公式为：

    如果 reduction = "none":

    .. math::

        output = |Target - Input|

    如果 reduction = "mean":

    .. math::

        output = \frac{1}{n}\sum_{i=1}^n|Target_i - Input_i|

    如果 reduction = "sum":

    .. math::

        output = \sum_{i=1}^n|Target_i - Input_i|

    参数:
        - **input** (Tensor): 输入张量。
        - **target** (Tensor): 目标张量。
        - **reduction** (str): 简化类型，可以为 ``"none"`` 、 ``"mean"`` 、 ``"sum"`` 。默认为 "mean" 。

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.tensor([[1, 1, 1], [2, 2, 2], [7, 7, 7]], dtype = flow.float32)
        >>> target = flow.tensor([[4, 4, 4], [4, 4, 4], [4, 4, 4]], dtype = flow.float32)
        >>> m = flow.nn.L1Loss(reduction="none")
        >>> out = m(input, target)
        >>> out
        tensor([[3., 3., 3.],
                [2., 2., 2.],
                [3., 3., 3.]], dtype=oneflow.float32)
        >>> m_mean = flow.nn.L1Loss(reduction="mean")
        >>> out = m_mean(input, target)
        >>> out
        tensor(2.6667, dtype=oneflow.float32)
        >>> m_mean = flow.nn.L1Loss(reduction="sum")
        >>> out = m_mean(input, target)
        >>> out
        tensor(24., dtype=oneflow.float32)
    """
)

reset_docstr(
    oneflow.nn.LayerNorm,
    r"""LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True)
    
    对小批量输入应用层归一化 (Layer Normalization) ，行为如论文 `Layer Normalization <https://arxiv.org/abs/1607.06450>`__ 所述。

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    在特定数字维度上分别计算均值和标准差，这些维度的形状必须由:attr:`normalized_shape` 指定。
    如果 :attr:`elementwise_affine` 为 `True` ，则 :math:`\gamma` 和 :math:`\beta` 是
    参数 :attr:`normalized_shape` 的可学习仿射变换参数。标准差是通过有偏估计器 (biased estimator) 计算的。


    .. note::
        与批量归一化 (Batch Normalization) 和实例归一化 (Instance Normalization) 使用 :attr:`affine` 
        选项为每个完整通道/平面应用标量 scale 和偏差不同，层归一化 (Layer Normalization) 使用 :attr:`elementwise_affine` 
        处理每个元素的 scale 和偏差。

    该层在训练和评估模式下都使用从输入数据计算的统计信息。

    参数：
        - **normalized_shape** (int 或 list 或 oneflow.Size): 来自预期大小输入的输入形状

            .. math::
                [* \times \text{normalized_shape}[0] \times \text{normalized_shape}[1] \times \ldots \times \text{normalized_shape}[-1]]

            如果使用单个整数，则将其视为单例列表，并且此模块将对最后一个维度进行标准化，该维度预计具有该特定大小。

        - **eps** (float, 可选的): 为数值稳定性而添加到分母的值。默认：1e-5
        - **elementwise_affine** (bool, 可选的): 如果为 ``True`` ，该模块具有可学习的逐元素仿射参数，
            并且将他们初始化为 1 （对于权重）和 0（对于偏差）。默认值： ``True``

    形状：
        - **Input** : :math:`(N, *)`
        - **Output** : :math:`(N, *)` （形状与输入相同）

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> x = flow.tensor([[[[-0.16046895, -1.03667831], [-0.34974465, 0.26505867]],[[-1.24111986, -0.53806001], [1.72426331, 0.43572459]],],[[[-0.77390957, -0.42610624], [0.16398858, -1.35760343]],[[1.07541728, 0.11008703], [0.26361224, -0.48663723]]]], dtype=flow.float32)
        >>> m = flow.nn.LayerNorm(2)
        >>> m(x)# doctest: +NORMALIZE_WHITESPACE
        tensor([[[[ 1.0000, -1.0000],
                  [-0.9999,  0.9999]],
        <BLANKLINE>
                 [[-1.0000,  1.0000],
                  [ 1.0000, -1.0000]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[-0.9998,  0.9998],
                  [ 1.0000, -1.0000]],
        <BLANKLINE>
                 [[ 1.0000, -1.0000], 
                  [ 1.0000, -1.0000]]]], dtype=oneflow.float32,
               grad_fn=<broadcast_add_backward>)
    """

)