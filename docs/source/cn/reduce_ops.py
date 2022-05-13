import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.max,
    r"""max(input, dim=None, keepdim=False) -> Tensor

    返回 :attr:`input` 中的最大值。
    
    参数：
        - **input** (oneflow.tensor): 输入张量
        - **dim** (int, optional): 要进行计算的维度。默认： `None`
        - **keepdim** (bool, optional): 输出张量是否保留 :attr:`input` 的维度。默认： `False`

    返回类型：
        张量或元组（oneflow.tensor, oneflow.tensor(dtype=int64)）：
        如果参数 :attr:`dim` 是 `None` ，返回 :attr:`input` 所有元素中的最大值。如果 :attr:`dim` 不是 `None` ，
        则返回一个包含张量的元组(values, indices)， `values` 是 :attr:`input` 当前维度的最大值，
        `indices` 是最大值在 :attr:`input` 中当前维度的索引。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.Tensor([[4, 1, 5], [2, 6, 3]])
        >>> flow.max(input)
        tensor(6., dtype=oneflow.float32)
        >>> (values, indices) = flow.max(input, dim=1)
        >>> values
        tensor([5., 6.], dtype=oneflow.float32)
        >>> indices
        tensor([2, 1], dtype=oneflow.int64)

    """
)

reset_docstr(
    oneflow.mean,
    r"""mean(input, dim=None, keepdim=False) -> Tensor
    
    计算给定维度上张量中各行元素的均值，如果 :attr:`dim` 为 None ，则计算所有元素的均值。

    参数：
        - **input** (oneflow.tensor): 输入张量
        - **dim** (int, optional): 要进行计算的维度。默认： `None`
        - **keepdim** (bool, optional): 输出张量是否保留 :attr:`input` 的维度。默认： `False`

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.Tensor([[1, 2, 3], [4, 5, 6]])
        >>> flow.mean(input)
        tensor(3.5000, dtype=oneflow.float32)
        >>> flow.mean(input, dim=0)
        tensor([2.5000, 3.5000, 4.5000], dtype=oneflow.float32)
        >>> flow.mean(input, dim=1)
        tensor([2., 5.], dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.min,
    r"""min(input, dim=None, keepdim=False) -> Tensor

    返回 :attr:`input` 中的最小值。
    
    参数：
        - **input** (oneflow.tensor): 输入张量
        - **dim** (int, optional): 要进行计算的维度。默认： `None`
        - **keepdim** (bool, optional): 输出张量是否保留 :attr:`input` 的维度。默认： `False`

    返回类型：
        张量或元组（oneflow.tensor, oneflow.tensor(dtype=int64)）：
        如果参数 :attr:`dim` 是 `None` ，返回 :attr:`input` 所有元素中的最小值。如果 :attr:`dim` 不是 `None` ，
        则返回一个包含张量的元组(values, indices)， `values` 是 :attr:`input` 当前维度的最小值，
        `indices` 是最小值在 :attr:`input` 中当前维度的索引。
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.Tensor([[4, 1, 5], [2, 6, 3]])
        >>> flow.min(input)
        tensor(1., dtype=oneflow.float32)
        >>> (values, indices) = flow.min(input, dim=1)
        >>> values
        tensor([1., 2.], dtype=oneflow.float32)
        >>> indices
        tensor([1, 0], dtype=oneflow.int64)

    """
)

reset_docstr(
    oneflow.prod,
    r"""prod(input, dim=None, keepdim=False) -> Tensor
    
    在给定维度上计算 :attr:`input` 中每行中元素的乘积，并返回一个包含结果的新 tensor 。
    
    注意： `如果参数 dim 为 None ，返回一个只有一个元素的 tensor ，其元素为 input 中所有数之积`

    参数：
        - **input** (Tensor): 输入源张量
        - **dim** (int): 要做乘法的维度

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.Tensor([[1, 2, 3], [4, 5, 6]])
        >>> flow.prod(input)
        tensor(720., dtype=oneflow.float32)
        >>> flow.prod(input, dim=0)
        tensor([ 4., 10., 18.], dtype=oneflow.float32)
        >>> flow.prod(input, dim=1)
        tensor([  6., 120.], dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.sum,
    r"""sum(input, dim=None, keepdim=False) -> Tensor
    
    在给定的维度 :attr:`dim` 计算 :attr:`input` 每列的元素和。如果没有设定 :attr:`dim` ，则会计算 :attr:`input` 所有元素的和。
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.Tensor([[1, 2, 3], [4, 5, 6]])
        >>> flow.sum(input)
        tensor(21., dtype=oneflow.float32)
        >>> flow.sum(input, dim=0)
        tensor([5., 7., 9.], dtype=oneflow.float32)
        >>> flow.sum(input, dim=1)
        tensor([ 6., 15.], dtype=oneflow.float32)
"""
)

reset_docstr(
    oneflow.all,
    r"""
    oneflow.all(input, dim=None, keepdim=False) -> Tensor

    对于给定维度 `dim` 的 `input` 的每一行，如果该行的所有元素都为 True，则返回 True，否则返回 False。如果维度是 None，则计算输入张量中的所有元素是否为真。
    
    如果 `keepdim` 为 `True` ，输出张量与 `input` 大小相同，但 `dim` 的维度大小为 1 的情况除外。否则， `dim` 会被压缩 :func:`oneflow.squeeze()` ，导致输出张量的大小减少 1（或 `len(dim)`）。

    参数:
        - **input** (oneflow.Tensor) - 输入张量
        - **dim** (int, optional) - 减少的维度。 默认值: `None`
        - **keepdim** (bool, optional) - 输出张量是否有 dim 的保留. 默认值: `False`

    示例:

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.Tensor([[1, 2, 3], [4, 5, 6]]) < 4
        >>> input
        tensor([[ True,  True,  True],
                [False, False, False]], dtype=oneflow.bool)
        >>> flow.all(input)
        tensor(False, dtype=oneflow.bool)
        >>> flow.all(input, 1)
        tensor([ True, False], dtype=oneflow.bool)
        >>> flow.all(input, 1, True)
        tensor([[ True],
                [False]], dtype=oneflow.bool)
"""
)

reset_docstr(
    oneflow.any,

    r"""oneflow.any(input, dim=None, keepdim=False) -> Tensor

    对于给定维度 `dim` 中的每一行输入，如果该行中的任何元素评估为 True，则返回 True，否则返回 False。如果维度为 None，则计算输入张量中的元素是否都为真。

<<<<<<< HEAD
    如果 `keepdim` 为 True，则输出张量的大小与输入相同，但尺寸为1的维度 `dim` 除外。否则，`dim` 被挤压 :func:`oneflow.squeeze()` ，导致输出张量具有1（或 `len(dim)`）更少的维度。
=======
    如果 `keepdim` 为 True，则输出张量的大小与输入相同，但维度 `dim` 大小为 1 的情况除外。否则，`dim` 会被压缩 :func: `oneflow.squeeze()` ，导致输出张量具有 1（或 `len(dim)`）更少的维度。
>>>>>>> c0d1838df21ce6972d2ab89d2efbcd83aff9dcc7

    参数:
        - **input** (oneflow.Tensor) - 输入张量
        - **dim** (int, optional) - 减少的维度。 默认值: `None`
        - **keepdim** (bool, optional) - 输出张量是否有 dim 的保留. 默认值: `False`

    示例:

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.Tensor([[1, 2, 3], [4, 5, 6]]) < 4
        >>> input
        tensor([[ True,  True,  True],
                [False, False, False]], dtype=oneflow.bool)
        >>> flow.any(input)
        tensor(True, dtype=oneflow.bool)
        >>> flow.any(input, 0)
        tensor([True, True, True], dtype=oneflow.bool)
        >>> flow.any(input, 0, True)
        tensor([[True, True, True]], dtype=oneflow.bool)
    """,

)
