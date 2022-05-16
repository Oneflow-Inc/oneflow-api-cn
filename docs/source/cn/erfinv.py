import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.erfinv,
    r"""计算 :attr: `input` 的反误差函数。反误差函数在 `(-1, 1)` 的范围内定义为:

    .. math::
        \mathrm{erfinv}(\mathrm{erf}(x)) = x

    参数:
        - **input**: (oneflow.Tensor) - 输入张量

    示例:

    .. code-block:: python
    
        >>> import oneflow as flow
        >>> import numpy as np
               
        >>> input=flow.tensor(np.random.randn(3,3).astype(np.float32))
        >>> of_out=flow.erfinv(input)
        >>> of_out.shape
        oneflow.Size([3, 3])

    """
)