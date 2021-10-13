import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.meshgrid,
    r"""meshgrid(*tensors) -> sequence of Tensors

    此接口与 PyTorch 一致。
    文档参考自：
    https://pytorch.org/docs/stable/_modules/torch/functional.html#meshgrid
    
    取 :math:`N` 个 tensor ，可以是标量或者 1 维张量，并创建 :math:`N` 个 N-D 网格，
    第 i 个网格是通过扩展第 i 个输入并且其维度由其他的输入决定。

    参数：
        **tensors** (list of Tensor): 标量或一维张量列表。标量将被自动视为大小为 :math:`(1,)` 的张量

    返回类型：
        **seq** (sequence of Tensors): 如果输入有 :math:`k` 个张量，大小分别为 :math:`(N_1,), (N_2,), ... , (N_k,)` ，
        则输出也有 :math:`k` 个张量，其中所有张量的大小为 :math:`(N_1, N_2, ... , N_k)` 。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input1 = flow.tensor([1, 2, 3], dtype=flow.float32)
        >>> input2 = flow.tensor([4, 5, 6], dtype=flow.float32)
        >>> of_x, of_y = flow.meshgrid(input1, input2)
        >>> of_x
        tensor([[1., 1., 1.],
                [2., 2., 2.],
                [3., 3., 3.]], dtype=oneflow.float32)
        >>> of_y
        tensor([[4., 5., 6.],
                [4., 5., 6.],
                [4., 5., 6.]], dtype=oneflow.float32)
    
    """
)
