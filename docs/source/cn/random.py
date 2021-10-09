import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.bernoulli,
    r"""
    bernoulli(x, *, generator=None, out=None)
    
    返回一个Tensor其参数为带有来自伯努利分布的二进制随机数(0 / 1) 。

    参数：
        - **input** (Tensor): the input tensor of probability values for the Bernoulli distribution
        - **generator** (Generator, optional) a pseudorandom number generator for sampling
        - **out** (Tensor, optional): the output tensor.

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