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


        >>> import oneflow as flow
        
        >>> m = flow.nn.Hardtanh()
        >>> x = flow.tensor([0.2, 0.3, 3.0, 4.0], dtype=flow.float32)
        >>> out = m(x)
        >>> out
        tensor([0.2000, 0.3000, 1.0000, 1.0000], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.mish,
    r"""
    mish(x: Tensor) -> Tensor 

    应用此 element-wise 公式：

    .. math::
        \text{mish}(x) = x * \text{tanh}(\text{softplus}(x))


    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([1, 2, 3], dtype=flow.float32)       

        >>> out = flow.mish(input)
        >>> out
        tensor([0.8651, 1.9440, 2.9865], dtype=oneflow.float32)

    更多细节参考 :class:`~oneflow.nn.Mish` 。
    
    """
)

reset_docstr(
    oneflow.nn.Mish,
    """逐元素地应用如下公式：

    .. math::
        \\text{Mish}(x) = x * \\text{Tanh}(\\text{Softplus}(x))

    .. note::
        请参考 `Mish: A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_ 。

    形状：
        - Input: :math:`(N, *)` ，其中 `*` 表示任意数量的附加维度。
        - Output: :math:`(N, *)` ，与输入的形状相同。

    示例：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = np.array([1, 2, 3]).astype(np.float32)
        >>> input = flow.Tensor(x)
        >>> mish = flow.nn.Mish()

        >>> out = mish(input)
        >>> out
        tensor([0.8651, 1.9440, 2.9865], dtype=oneflow.float32)
    """
)

reset_docstr(
    oneflow.tanh,
    r"""
    tanh(x) -> Tensor 

    公式为：

    .. math::

        out = \frac{e^x-e^{-x}}{e^x+e^{-x}}

    更多信息请参考 :class:`~oneflow.nn.Tanh` 。
    """,
)

reset_docstr(
    oneflow.relu,
    r"""relu(input, inplace) -> Tensor

    对 :attr:`input` 逐元素应用 ReLU 函数(Rectified Linear Unit，线性整流函数)。更多信息请参考 :class:`~oneflow.nn.ReLU` 。

    参数：
        - **input** (Tensor)
        - **inplace** (Bool): 如果为 ``True`` ，则以原地算法执行此算子。默认为 ``False``
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([1, -2, 3], dtype=flow.float32)
        >>> output = flow.relu(input)
        >>> output
        tensor([1., 0., 3.], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.nn.functional.conv1d,
    r"""
    conv1d(input, weight, bias=None, stride=[1], padding=[0], dilation=[1], groups=1) -> Tensor

    文档引用自: https://pytorch.org/docs/stable/generated/torch.nn.functional.conv1d.html?highlight=conv1d

    对由多个输入平面组成的输入信号应用一维卷积。

    请参阅: :class:`~oneflow.nn.Conv1d` 获取有关详细信息和输出形状。

    参数：
        - **input**: 形状的量化输入张量: :math:`(\text{minibatch} , \text{in_channels} , iW)` 
        - **weight**: 形状的量化滤波器: :math:`(\text{out_channels} , \frac{\text{in_channels}}{\text{groups}} , iW)` 
        - **bias**: 非量化的形状的偏置张量: :math:`(\text{out_channels})` 。张量类型必须为 `flow.float` 。
        - **stride**: 卷积核的步长。可以是单个数字或元组 `(sW,)` 。 默认值: 1
        - **padding**: 输入两侧的隐式填充。可以是单个数字或元组 `(padW,)` 。 默认值: 0
        - **dilation**: 内核元素之间的间距。可以是单个数字或元组 `(dW,)` 。 默认值: 1
        - **groups**: 将输入分成组: :math:`\text{in_channels}` 应该可以被组数整除。默认值: 1

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import oneflow.nn as nn
        
        >>> input = flow.randn(33, 16, 30, dtype=flow.float32)
        >>> filters = flow.randn(20, 16, 5, dtype=flow.float32)
        >>> out = nn.functional.conv1d(input, filters,stride=[1], padding=[0], dilation=[1], channel_pos="channels_first")
        
    """

)
