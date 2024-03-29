import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.bernoulli,
    r"""
    bernoulli(x, *, generator=None, out=None)
    
    返回一个 Tensor，起元素为0或1，且符合伯努利分布 。

    参数：
        - **x** (Tensor)： 伯努利分布的概率值的输入张量
        - **generator** (Generator, 可选)： 用于采样的伪随机数生成器
        - **out** (Tensor, 可选): 输出张量

    形状：
        - **Input** :math:`(*)`： 输入可以是任何形状
        - **Output** :math:`(*)`： 输出与输入的形状相同

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

