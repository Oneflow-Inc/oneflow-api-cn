import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.view,
    r"""view(input, *shape) -> Tensor
    
    此接口与 PyTorch 一致。 文档参考自：https://pytorch.org/docs/stable/generated/torch.Tensor.view.html

    
    返回一个新的 tensor ，其数据与 :attr:`input` 相同，但形状 :attr:`shape` 不同。

    返回的 tensor 与 :attr:`input` 共享相同的数据并且必须具有相同数量的元素，但是形状可以不同。
    对于要被查看的 tensor ，新的视图大小必须与其原始大小和 step 兼容，每个新视角的维度必须为原始维度的子空间，
    或者为跨越原始维度 :math:`d, d+1, \dots, d+k` 的 span 满足以下类似邻接条件 :math:`\forall i = d, \dots, d+k-1` 。

    .. math::

      \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]

    
    否则将无法以视图查看  为形状 :attr:`shape` 且不复制 :attr:`input` （例如通过 :meth:`contiguous`）。
    当不清楚是否可以执行 :meth:`view` 时，建议使用 :meth:`reshape` ，因为 :meth:`reshape` 在兼容的时候返回
    :attr:`input` 的视图，不兼容的时候复制 :attr:`input` （相当于调用 :meth:`contiguous` ）。

    参数：
        - **input** (Tensor)
        - **shape** (flow.Size 或 int...)

    返回类型：
        与 :attr:`input` 数据类型相同的 tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=flow.float32)

        >>> y = input.view(2, 2, 2, -1).shape
        >>> y
        oneflow.Size([2, 2, 2, 2])

    """

)
