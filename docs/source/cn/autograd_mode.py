import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.no_grad,
    r"""
    禁用梯度计算上下文管理器。

    当确定不调用 Tensor.backward() 时，禁用梯度计算对于推理很有用，此操作相比 requires_grad=True 
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
