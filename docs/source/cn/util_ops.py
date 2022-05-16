import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.isinf,
    r"""
    测试输入的每个元素是否为 infinite (正或负的 infinite)

    参数：
        - **input** (Tensor) - 输入张量。
    
    返回类型：
        一个布尔类型的张量，当输入为 infinite 为 True，否则为 False。

    示例：

        >>> import oneflow as flow
        >>> flow.isinf(flow.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
        tensor([False,  True, False,  True, False], dtype=oneflow.bool)
    """
)

reset_docstr(
    oneflow.isnan,
    r"""
    返回一个带有布尔元素的新张量，表示输入的每个元素是否为 NaN。

    参数：
        - **input** (Tensor) - 输入张量。
    
    返回类型：
        一个布尔张量，在输入为 NaN 的情况下为 True，否则为 False。

    示例：

        >>> import oneflow as flow
        >>> flow.isnan(flow.tensor([1, float('nan'), 2]))
        tensor([False,  True, False], dtype=oneflow.bool)
    """
)