import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.ones_like,
    r"""ones_like(x) -> Tensor

    返回一个元素全部为值为 1 的标量，且形状与 `x` 相同的 tensor。
    flow.ones_like(x) 等价于 flow.ones(x.shape, dtype=x.dtype)

    参数：
        **x** (Tensor): 输入的形状将会决定输出的形状

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.randn(5, dtype=flow.float32)
        >>> y = flow.ones_like(x)
        >>> y
        tensor([1., 1., 1., 1., 1.], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.zeros_like,
    r"""zeros_like(x) -> Tensor

    返回一个元素全部为值为 0 的标量，形状和 `x` 相同的 Tensor。
    flow.zeros_like(x) 等价于 flow.zeros(x.shape, dtype=x.dtype)

    参数：
        **x** (Tensor): 输入的形状将决定输出的形状

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.randn(5, dtype=flow.float32)
        >>> y = flow.zeros_like(x)
        >>> y
        tensor([0., 0., 0., 0., 0.], dtype=oneflow.float32)

    """,
)
