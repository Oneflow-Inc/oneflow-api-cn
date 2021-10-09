import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.bernoulli,
    r"""
    bernoulli(x, *, generator=None, out=None)
    
    返回一个Tensor其参数为带有来自伯努利分布的二进制随机数(0 / 1) 。

    参数：
        - **x** (Tensor)： 伯努利分布的概率值的输入张量
        - **generator** (Generator, optional)： 用于采样的伪随机数生成器
        - **out** (Tensor, optional): 输出张量

    形状：
        - **Input** :math:`(*)`. Input can be of any shape
        - **Output** :math:`(*)`. Output is of the same shape as input

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> x = flow.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=flow.float32)
        >>> y = flow.bernoulli(x)
        >>> y
        tensor([[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.Tensor.atan2,
    r"""
    参考 :func:`oneflow.atan2`
    """,
)

reset_docstr(
    oneflow.Tensor.numel,
    r"""
    参考 :func:`oneflow.numel`
    """,
)

reset_docstr(    
    oneflow.Tensor.transpose,
    r"""
    参考 :func:`oneflow.transpose`
    """,
    )
