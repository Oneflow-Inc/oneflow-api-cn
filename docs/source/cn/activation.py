import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.nn.ReLU,
    r"""ReLU(inplace=False)
    
    ReLU 激活函数，对张量中的每一个元素做 element-wise 运算，公式如下:

    :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`

    参数:
        inplace: 是否做 in-place 操作。 默认为 ``False``

    形状:
        - Input: :math:`(N, *)` 其中 `*` 的意思是，可以指定任意维度
        - Output: :math:`(N, *)` 输入形状与输出形状一致

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> relu = flow.nn.ReLU()
        >>> x = flow.tensor([1, -2, 3], dtype=flow.float32)
        >>> relu(x)
        tensor([1., 0., 3.], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.nn.Hardtanh,
    r""" 

    按照以下公式（HardTanh），进行 element-wise 操作：

    .. math::
        \text{HardTanh}(x) = \begin{cases}
            1 & \text{ if } x > 1 \\
            -1 & \text{ if } x < -1 \\
            x & \text{ otherwise } \\
        \end{cases}

    默认的线性范围为 :math:`[-1, 1]`，可以通过设置参数
    :attr:`min_val` 和 :attr:`max_val` 改变。

    参数:
        - min_val: 线性范围的下界。 默认值: -1
        - max_val: 线性范围的上界。 默认值: 1
        - inplace: 是否做 in-place 操作。默认为 ``False``

    因为有了参数 :attr:`min_val` 和 :attr:`max_val`，原有的
    参数 :attr:`min_value` and :attr:`max_value` 已经被不再推荐使用。

    形状:
        - Input: :math:`(N, *)` 其中 `*` 的意思是，可以指定任意维度
        - Output: :math:`(N, *)` 输入形状与输出形状一致

    示例:

    .. code-block:: python


        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> m = flow.nn.Hardtanh()
        >>> arr = np.array([0.2, 0.3, 3.0, 4.0])
        >>> x = flow.Tensor(arr)
        >>> out = m(x)
        >>> out
        tensor([0.2000, 0.3000, 1.0000, 1.0000], dtype=oneflow.float32)

    """,
)
