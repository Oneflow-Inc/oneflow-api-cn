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
        >>> out = nn.functional.conv1d(input, filters,stride=[1], padding=[0], dilation=[1])

    """,    
    
)

reset_docstr(
    oneflow.optim.Adagrad,
    r"""实现Adagrad优化器。

        公式是: 

        .. math:: 

            & S_{t} = S_{t-1} + grad \odot grad 
            
            & decay\_lr = \frac{learning\_rate}{(1 + (train\_step - 1) * lr\_decay)}

            & X_{t} = X_{t-1} - \frac{decay\_lr}{\sqrt{S_{t} + \epsilon}} \odot grad

        参数:
            - **params** (Union[Iterator[Parameter], List[Dict]]): 可迭代的参数 用于优化或字典定义。
            - **parameter** (groups)
            - **lr** (float, 可选的): 学习率. 默认为 0.001。
            - **lr_decay** (float, 可选的): 学习率的衰减因子. 默认为 0.0。
            - **weight_decay** (float, 可选的): 重量衰变. 默认为 0。
            - **initial_accumulator_value** (float, 可选的): S 的初值. 默认为 0.0。
            - **eps** (float, 可选的): 在分母上加的一个小常数值用于提高数值的稳定性. 默认为 1e-10。
        
        例如: 

        示例 1: 

        .. code-block:: python

            # 假设 net 是一个自定义模型. 
            adagrad = flow.optim.Adagrad(net.parameters(), lr=1e-3)

            for epoch in range(epochs):
                # 读取数据, 计算损失等等. 
                # ...
                loss.backward()
                adagrad.step()
                adagrad.zero_grad()

        示例 2: 

        .. code-block:: python 

            # 假设 net 是一个自定义模型. 
            adagrad = flow.optim.Adagrad(
                [
                    {
                        "params": net.parameters(),
                        "lr": learning_rate,
                        "clip_grad_max_norm": 0.5,
                        "clip_grad_norm_type": 2.0,
                    }
                ],
            )

            for epoch in range(epochs):
                # 读取数据, 计算损失等等. 
                # ...
                loss.backward()
                adagrad.clip_grad()
                adagrad.step()
                adagrad.zero_grad()

        如果想使用 clip_grad, 可以参考这个例子。 

        关于 `clip_grad_max_norm` 和 `clip_grad_norm_type` 更多的细节, 可以参考 :func:`oneflow.nn.utils.clip_grad_norm_`.  
        
    """
)
