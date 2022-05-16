import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.no_grad,
    r"""
    禁用梯度计算的上下文管理器。

    当确定不调用 Tensor.backward() 时，禁用梯度计算对于推理很有效，此操作相比 requires_grad=True 
    时可以减少计算的内存消耗。

    此模式下，任何计算都会视 requires_grad 为 False ，即便输入有 requires_grad=True。

    此上下文管理器只会影响本地线程，不会影响其他线程的运算。

    也起到装饰器的作用。 (请确保用括号进行实例化)

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.ones(2, 3, requires_grad=True)
        >>> with flow.no_grad():
        ...     y = x * x
        >>> y.requires_grad
        False
        >>> @flow.no_grad()
        ... def no_grad_func(x):
        ...     return x * x
        >>> y = no_grad_func(x)
        >>> y.requires_grad
        False
    
    """
)

reset_docstr(
    oneflow.set_grad_enabled,
    r"""
    启用梯度计算的上下文管理器。

    如果梯度计算通过 no_grad 被禁用，则启用它。

    这个上下文管理器是线程本地的；它不会影响其他线程的计算。

    也可以作为一个装饰器。(确保用括号来实例化)。

    参数：
        - **mode** (bool) - 标记是否启用或禁用梯度计算。（默认：True）
    
    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.ones(2, 3, requires_grad=True)
        >>> with flow.set_grad_enabled(True):
        ...     y = x * x
        >>> y.requires_grad
        True
        >>> @flow.set_grad_enabled(False)
        ... def no_grad_func(x):
        ...     return x * x
        >>> y = no_grad_func(x)
        >>> y.requires_grad
        False
    """
)