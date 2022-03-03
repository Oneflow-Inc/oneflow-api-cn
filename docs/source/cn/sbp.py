import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.sbp.sbp,
    r"""
    一个 sbp 是一个表示 `oneflow.Tensor` 在物理设备集群中的分布式数据类型的对象，表示全局视角下的 Tensor 与物理设备上的 Tensor 之间的映射关系。
    
    `sbp` 有三种类型：
    
    1. `split` 表示物理设备上的Tensor 是将全局视角下的 Tensor 切分得到的。切分时需要指定切分的维度。物理设备上的 Tensor 如果根据切分的维度被拼接，可以还原得到全局视角的 Tensor 。
    
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