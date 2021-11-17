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
