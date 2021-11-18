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
