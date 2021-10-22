import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.atan2,
    r"""atan2(input, other)

    考虑象限的 :math:`input_{i}/other_{i}`
    的按元素反正切。返回元素为向量 :math:`(other_{i},input_{i})`
    和向量 (1, 0) 之间的按元素夹角（以弧度表示并带符号）的新张量。

    `input` 和 `other` 的形状必须是可广播的。

    参数：
        - **input** (Tensor): 第一个输入张量。

        - **other** (Tensor): 第二个输入张量。

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> x1 = flow.tensor([1,2,3], dtype=flow.float32)
        >>> y1 = flow.tensor([3,2,1], dtype=flow.float32)
        >>> x2 = flow.tensor([1.53123589,0.54242598,0.15117185], dtype=flow.float32)
        >>> y2 = flow.tensor([-0.21906378,0.09467151,-0.75562878], dtype=flow.float32)
        >>> x3 = flow.tensor([1,0,-1], dtype=flow.float32)
        >>> y3 = flow.tensor([0,1,0], dtype=flow.float32)

        >>> flow.atan2(x1,y1)
        tensor([0.3218, 0.7854, 1.2490], dtype=oneflow.float32)
        >>> flow.atan2(x2,y2)
        tensor([1.7129, 1.3980, 2.9441], dtype=oneflow.float32)
        >>> flow.atan2(x3,y3)
        tensor([ 1.5708,  0.0000, -1.5708], dtype=oneflow.float32)
    """,
)
