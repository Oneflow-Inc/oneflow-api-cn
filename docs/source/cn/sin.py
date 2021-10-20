import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.sin,
    r"""sin(input) -> Tensor
    
    返回一个元素为 :attr:`input` 正弦值的新张量。
  
    .. math::
        \text{y}_{i} = \sin(\text{x}_{i})
    
    参数：
        **input** (Tensor): 输入张量
        
    示例：
    .. code-block:: python

        >>> import oneflow as flow
        >>> x1 = flow.tensor([-0.5461,  0.1347, -2.7266, -0.2746], dtype=flow.float32)
        >>> y1 = flow.sin(x1)
        >>> y1
        tensor([-0.5194,  0.1343, -0.4032, -0.2712], dtype=oneflow.float32)
        
        >>> x2 = flow.tensor([-1.4, 2.6, 3.7], dtype=flow.float32, device=flow.device('cuda'))
        >>> y2 = flow.sin(x2)
        >>> y2
        tensor([-0.9854,  0.5155, -0.5298], device='cuda:0', dtype=oneflow.float32)
        
    """,
)
