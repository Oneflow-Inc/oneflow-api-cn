import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.is_tensor,
    r"""
    is_tensor(input) -> (bool)

    注意，这个函数只是在执行 ``isinstance(obj, Tensor)`` 。使用 ``isinstance`` 检查对 mypy 的类型更明确，所以建议使用该函数而不是 ``is_tensor`` 。
    
    参数：
        obj (Object): 被测试的对象。
    
    示例:

    .. code-block:: python
    
        >>> import oneflow as flow

        >>> x=flow.tensor([1,2,3])
        >>> flow.is_tensor(x)
        True
    
    """
)