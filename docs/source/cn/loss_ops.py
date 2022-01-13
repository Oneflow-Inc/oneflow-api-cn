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
    oneflow.nn.BCELoss,
    r"""BCELoss(weight=None, reduction='mean') -> Tensor

    计算二值交叉熵损失 (binary cross-entropy loss)。

    公式为：

    如果 :attr:`reduction` = "none" ：

    .. math::

        out = -(Target_i*log(Input_i) + (1-Target_i)*log(1-Input_i))

    如果 :attr:`reduction` = "mean":

    .. math::

        out = -\frac{1}{n}\sum_{i=1}^n(Target_i*log(Input_i) + (1-Target_i)*log(1-Input_i))

    如果 :attr:`reduction` = "sum":

    .. math::

        out = -\sum_{i=1}^n(Target_i*log(Input_i) + (1-Target_i)*log(1-Input_i))

    参数：
        - **weight** (oneflow.Tensor, 可选的): 手动重新调整损失的权重。默认为 ``None`` ，对应的权重值为 1
        - **reduction** (str, 可选的): reduce 的方式，可以是 "none" 、 "mean" 、 "sum" 中的一种。默认为 "mean" 

    Attention:
        输入值必须在区间 (0, 1) 内。否则此损失函数可能返回 `nan` 值。

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.tensor([[1.2, 0.2, -0.3], [0.7, 0.6, -2]], dtype=flow.float32)
        >>> target = flow.tensor([[0, 1, 0], [1, 0, 1]], dtype=flow.float32)
        >>> weight = flow.tensor([[2, 2, 2], [2, 2, 2]], dtype=flow.float32)
        >>> activation = flow.nn.Sigmoid()
        >>> sigmoid_input = activation(input)
        >>> m = flow.nn.BCELoss(weight, reduction="none")
        >>> out = m(sigmoid_input, target)
        >>> out
        tensor([[2.9266, 1.1963, 1.1087],
                [0.8064, 2.0750, 4.2539]], dtype=oneflow.float32)
        >>> m_sum = flow.nn.BCELoss(weight, reduction="sum")
        >>> out = m_sum(sigmoid_input, target)
        >>> out
        tensor(12.3668, dtype=oneflow.float32)
        >>> m_mean = flow.nn.BCELoss(weight, reduction="mean")
        >>> out = m_mean(sigmoid_input, target)
        >>> out
        tensor(2.0611, dtype=oneflow.float32)
        >>> m_none = flow.nn.BCELoss()
        >>> out = m_none(sigmoid_input, target)
        >>> out
        tensor(1.0306, dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.nn.BCEWithLogitsLoss,
    r"""BCEWithLogitsLoss(weight=None, reduction='mean', pos_weight=None) -> Tensor
    
    此运算将 `Sigmoid` 和 `BCELoss` 组合在一起。为了数据的稳定性，我们用了一些数学技巧，而不是将 `Sigmoid` 作用于 `BCELoss` 层。

    公式为：

    如果 :attr:`reduction` = ``"none"``:

    .. math::

        out = -weight*[Pos\_weight*y*log\sigma({x}) + (1-y)*log(1-\sigma(x))]

    如果 :attr:`reduction` = ``"mean"``:

    .. math::

        out = -\frac{weight}{n}\sum_{i=1}^n[Pos\_weight*y*log\sigma({x}) + (1-y)*log(1-\sigma(x))]

    如果 :attr:`reduction` = ``"sum"``:

    .. math::

        out = -weight*\sum_{i=1}^n[Pos\_weight*y*log\sigma({x}) + (1-y)*log(1-\sigma(x))]

    参数：
        - **weight** (Tensor, 可选的): 手动重新调整损失的权重。默认为 ``None``
        - **reduction** (str, 可选的): reduce 的方式，可以是 ``"none"`` 、 ``"mean"`` 、 ``"sum"`` 中的一种。默认为 "mean" 。如果为 ``'none'`` 则不进行 reduce 。如果为 ``'mean'`` ，输出的值的和除以元素数。如果为 ``'sum'`` ，输出将被求和。默认为 ``"mean"``
        - **pos_weight** (Tensor, 可选的): 手动重新调整正例的权重。

    形状：
        - **Input** : :math:`(N,*)` 其中 `*` 的意思是，可以增加任意维度
        - **Target** : :math:`(N,*)` 与输入形状一样
        - **Output** : 标量。如果 :attr:`reduction` 为 ``"none"`` ，则 :math:`(N,*)` 和输入形状一样
        
    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([[1.2, 0.2, -0.3], [0.7, 0.6, -2], [0.7, 0.6, -2]], dtype=flow.float32)
        >>> target = flow.tensor([[0, 1, 0], [1, 0, 1], [1, 0, 1]], dtype=flow.float32)
        >>> weight = flow.tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]], dtype=flow.float32)
        >>> pos_weight = flow.tensor([1.2, 1.3, 1.4], dtype=flow.float32)

        >>> m = flow.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight, reduction="none")
        >>> out = m(input, target)
        >>> out
        tensor([[2.9266, 1.5552, 1.1087],
                [0.9676, 2.0750, 5.9554],
                [0.9676, 2.0750, 5.9554]], dtype=oneflow.float32)

        >>> m = flow.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight, reduction="mean")
        >>> out = m(input, target)
        >>> out
        tensor(2.6207, dtype=oneflow.float32)

        >>> m = flow.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight, reduction="sum")
        >>> out = m(input, target)
        >>> out
        tensor(23.5865, dtype=oneflow.float32)

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
    oneflow.nn.BatchNorm1d,
    r"""BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    
    在 2D 或 3D 输入（具有可选附加通道维度的小批量 1D 输入）上应用批归一化 (Batch Normalization) 。行为与论文 `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ 一致。


    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    按小批量逐维度求平均值和标准差， :math:`\gamma` 和 :math:`\beta` 是大小为 `C` 的可学习参数向量（ `C` 是输入的大小）。
    默认情况下，:math:`\gamma` 的元素均为 1 而 :math:`\beta` 的元素均为 0 。标准差的计算等价于 `torch.var(input, unbiased=False)` 。

    此外，默认情况下，在训练期间，该层不断估计计算的均值和方差，然后评估时将其归一化。运行估计默认 :attr:`momentum` 为 0.1 。

    如果 :attr:`track_running_stats` 被设置为 ``False`` ，则该层不会继续进行估计，并且在评估时也使用批处理统计信息。

    .. note::
        :attr:`momentum` 参数不同于优化器 (optimizer) 类中使用的参数或传统的动量概念。数学上，这里的更新规则是：
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t` ，其中 :math:`\hat{x}` 是估计的统计量， :math:`x_t` 是新的观察值。

    因为批归一化 (Batch Normalization) 是在 `C` 维度上完成的，计算 `(N, L)` 切片的统计数据，所以常称其为 Temporal Batch Normalization 。
    
    参数：
        - **num_features** : :math:`C` 来自于大小为 :math:`(N, C, L)` 的预期输入或 :math:`L` 来自大小为 :math:`(N, L)` 的输入
        - **eps** : 为数值稳定性而为分母加的值。默认为：1e-5
        - **momentum** : 用于 :attr:`running_mean` 和 :attr:`running_var` 计算的值。设定为 ``None`` 则计算移动平均 (Moving average) ，默认：0.1
        - **affine** : 如果为 ``True`` ，该模块具有可学习的仿射参数。默认为 ``True`` 
        - **track_running_stats** : 当设置为 ``True`` 时，该模块跟踪运行均值和方差，当设置为 ``False`` 时，此模块不会跟踪此类统计信息，
            并将统计缓冲区 :attr:`running_mean` 和 :attr:`running_var` 初始化为 ``None`` 。当这些缓冲区为“无”时，此模块在训练和评估模式中始终使用批处理统计信息。默认值： ``True``
    
    形状：
        - **Input** : :math:`(N, C)` 或 :math:`(N, C, L)`
        - **Output** : :math:`(N, C)` 或 :math:`(N, C, L)` （与输入形状相同）

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> x = flow.randn(20, 100)
        >>> m = flow.nn.BatchNorm1d(100)
        >>> y = m(x)

    
    """
)

reset_docstr(
    oneflow.nn.BatchNorm2d,
    r"""BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    在 4D 输入（具有可选附加通道维度的小批量 2D 输入）上应用批归一化 (Batch Normalization) 。行为与论文 `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ 一致。

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    按小批量逐维度求平均值和标准差， :math:`\gamma` 和 :math:`\beta` 是大小为 `C` 的可学习参数向量（ `C` 是输入的大小）。
    默认情况下，:math:`\gamma` 的元素均为 1 而 :math:`\beta` 的元素均为 0 。标准差的计算等价于 `torch.var(input, unbiased=False)` 。

    此外，默认情况下，在训练期间，该层不断估计计算的均值和方差，然后评估时将其归一化。运行估计默认 :attr:`momentum` 为 0.1 。

    如果 :attr:`track_running_stats` 被设置为 ``False`` ，则该层不会继续进行估计，并且在评估时也使用批处理统计信息。

    .. note::
        :attr:`momentum` 参数不同于优化器 (optimizer) 类中使用的参数或传统的动量概念。数学上，这里的更新规则是：
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t` ，其中 :math:`\hat{x}` 是估计的统计量， :math:`x_t` 是新的观察值。

    因为批归一化 (Batch Normalization) 是在 `C` 维度上完成的，计算 `(N, H, W)` 切片的统计数据，所以常称其为 Spatial Batch Normalization 。

    参数：
        - **num_features** : :math:`C` 来自于大小为 :math:`(N, C, H, W)` 的预期输入
        - **eps** : 为数值稳定性而为分母加的值。默认为：1e-5
        - **momentum** : 用于 :attr:`running_mean` 和 :attr:`running_var` 计算的值。设定为 ``None`` 则计算移动平均 (Moving average) ，默认：0.1
        - **affine** : 如果为 ``True`` ，该模块具有可学习的仿射参数。默认为 ``True`` 
        - **track_running_stats** : 当设置为 ``True`` 时，该模块跟踪运行均值和方差，当设置为 ``False`` 时，此模块不会跟踪此类统计信息，
            并将统计缓冲区 :attr:`running_mean` 和 :attr:`running_var` 初始化为 ``None`` 。当这些缓冲区为“无”时，此模块在训练和评估模式中始终使用批处理统计信息。默认值： ``True``

    形状：
        - **Input** : :math:`(N, C, H, W)` 
        - **Output** : :math:`(N, C, H, W)` （与输入形状相同）

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> x = flow.randn(4, 2, 8, 3)
        >>> m = flow.nn.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1)
        >>> y = m(x)

    """
)

reset_docstr(
    oneflow.nn.BatchNorm3d,
    r"""BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    在 5D 输入（具有可选附加通道维度的小批量 3D 输入）上应用批归一化 (Batch Normalization) 。行为与论文 `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ 一致。

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    按小批量逐维度求平均值和标准差， :math:`\gamma` 和 :math:`\beta` 是大小为 `C` 的可学习参数向量（ `C` 是输入的大小）。
    默认情况下，:math:`\gamma` 的元素均为 1 而 :math:`\beta` 的元素均为 0 。标准差的计算等价于 `torch.var(input, unbiased=False)` 。

    此外，默认情况下，在训练期间，该层不断估计计算的均值和方差，然后评估时将其归一化。运行估计默认 :attr:`momentum` 为 0.1 。

    如果 :attr:`track_running_stats` 被设置为 ``False`` ，则该层不会继续进行估计，并且在评估时也使用批处理统计信息。

    .. note::
        :attr:`momentum` 参数不同于优化器 (optimizer) 类中使用的参数或传统的动量概念。数学上，这里的更新规则是：
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t` ，其中 :math:`\hat{x}` 是估计的统计量， :math:`x_t` 是新的观察值。
    
    因为批归一化 (Batch Normalization) 是在 `C` 维度上完成的，计算 `(N, H, W)` 切片的统计数据，所以常称其为 Spatial Batch Normalization 。

    参数：
        - **num_features** : :math:`C` 来自于大小为 :math:`(N, C, D, H, W)` 的预期输入
        - **eps** : 为数值稳定性而为分母加的值。默认为：1e-5
        - **momentum** : 用于 :attr:`running_mean` 和 :attr:`running_var` 计算的值。设定为 ``None`` 则计算移动平均 (Moving average) ，默认：0.1
        - **affine** : 如果为 ``True`` ，该模块具有可学习的仿射参数。默认为 ``True`` 
        - **track_running_stats** : 当设置为 ``True`` 时，该模块跟踪运行均值和方差，当设置为 ``False`` 时，此模块不会跟踪此类统计信息，
            并将统计缓冲区 :attr:`running_mean` 和 :attr:`running_var` 初始化为 ``None`` 。当这些缓冲区为“无”时，此模块在训练和评估模式中始终使用批处理统计信息。默认值： ``True``
    
    形状：
        - **Input** : :math:`(N, C, D, H, W)` 
        - **Output** : :math:`(N, C, D, H, W)` （与输入形状相同）

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> x = flow.randn(3, 2, 5, 8, 4)
        >>> m = flow.nn.BatchNorm3d(num_features=2, eps=1e-5, momentum=0.1)
        >>> y = m(x)
        >>> y.size()
        oneflow.Size([3, 2, 5, 8, 4])

    """
)

reset_docstr(
    oneflow.nn.CELU,
    r"""CELU(alpha=1.0, inplace=False)

    应用逐元素方程：

    .. math::

        \text{CELU}(x, \alpha) = \begin{cases}
				x & \text{ if } x \ge 0  \\
                \alpha*(exp(\frac{x}{\alpha})-1) & \text{ otherwise } \\
    		    \end{cases}

    参数：
        - **alpha** (float): CELU 公式中的 :math:`\alpha` 。默认值：1.0
        - **inplace** (bool): 是否执行 place 操作。默认： ``False``

    形状：
        - **Input** : :math:`(N,*)` 其中 `*` 的意思是，可以增加任意维度
        - **Output** : :math:`(N, *)`, 与输入相同

    示例：

    .. code-block:: python


        >>> import oneflow as flow
        
        >>> input = flow.tensor([-0.5, 0, 0.5], dtype=flow.float32)
        >>> celu = flow.nn.CELU(alpha=0.5)

        >>> out = celu(input)
        >>> out
        tensor([-0.3161,  0.0000,  0.5000], dtype=oneflow.float32)

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
        >>> m(x)
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
