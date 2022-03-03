import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.sbp.sbp,
    r"""
    一个 sbp 是一个 Tensor 在物理设备集群中的分布式数据类型的对象，表示全局视角下的 Tensor 与物理设备上的 Tensor 之间的映射关系。
    
    `sbp` 有三种类型：
    
    1. :attr:`split` 表示物理设备上的 Tensor 是将全局视角下的 Tensor 切分得到的。切分时需要指定切分的维度。物理设备上的 Tensor 如果根据切分的维度被拼接，可以还原得到全局视角的 Tensor 。
    
    2. :attr:`broadcast` 表示物理设备上的 Tensor 是全局视角下的 Tensor 完全相同的拷贝。
    
    3. :attr:`partial_sum` 表示全局视角下的 Tensor 与物理设备上的 Tensor 的形状相同，但是物理设备上的值，只是全局视角下 Tensor 的一部分。返回的全局视角下的 Tensor 是通过把物理设备上的 Tensor 按照对应的位置（按元素）相加得到的。
    
    一个 Tensor的 :attr:`sbp` 可以通过 Tensor.sbp 属性访问。
    
    一个 sbp 可以以多种方式被构造：
    
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