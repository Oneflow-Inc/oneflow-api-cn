import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.sbp.sbp,
    r"""
    sbp的中文文档
    A sbp is an object representing the distribution type of a oneflow.Tensor around the device group,
    which represents the mapping relationship between the logical Tensor and the physical Tensor.
    一个 sbp 是一个表示 `oneflow.Tensor` 在物理设备集群中的分布式数据类型的对象，表示逻辑张量与物理张量之间的映射关系。
    
    `sbp` 有三种类型：
    
    1. split: 
        Indicates that the physical Tensors are obtained by splitting the logical Tensor.
        Split will contain a parameter Axis, which represents the dimension to be split.
        If all the physical Tensors are spliced according to the dimensions of Split,
        the logical Tensor can be restored.
    
    2. broadcast: 
        Indicates that the physical Tensors are copies of the logical Tensor, which are
        exactly the same.
    
    3. partial_sum: 
        Indicates that the physical Tensors have the same shape as the logical Tensor,
        but the value of the element at each corresponding position is a part of the
        value of the element at the corresponding position of the logical Tensor. The
        logical Tensor can be returned by adding all the physical Tensors according to
        the corresponding positions (element-wise).
    
    A oneflow.Tensor's sbp can be accessed via the Tensor.sbp property.
    
    A sbp can be constructed in several ways:
    
    .. code-block:: python

        >>> import oneflow as flow
        
        >>> s = flow.sbp.split(0)
        >>> s
        oneflow.sbp.split(axis=0)
        >>> b = flow.sbp.broadcast
        >>> b
        oneflow.sbp.broadcast
        >>> p = flow.sbp.partial_sum
        >>> p
        oneflow.sbp.partial_sum
    
    """
)