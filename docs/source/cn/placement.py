import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.env.all_device_placement,
    r"""
    返回 env 下所有的机器的所有设备的 placement。

    参数：
        - **device_type** (str): cuda 或者 cpu

    示例：

    .. code-block:: python

        # world_size = 4, node_size = 1
        import oneflow as flow
        
        p = flow.env.all_device_placement("cuda") # oneflow.placement(device_type="cuda", machine_device_ids={0 : [0, 1, 2, 3]}, hierarchy=(4,))
        p = flow.env.all_device_placement("cpu") # oneflow.placement(device_type="cpu", machine_device_ids={0 : [0, 1, 2, 3]}, hierarchy=(4,))

    """
)

reset_docstr(
    oneflow.placement,
    r"""
    oneflow.placement 是一个对象，用于指代一个 oneflow.Tensor 被分配或即将被分配到的设备组。oneflow.placement 包含一个设备类型 ('cpu' 或者 'cuda') 和对应的设备序列。

    oneflow.Tensor 的 placement 可以通过 Tensor.placement 访问。

    oneflow.placement 可以通过以下几种方式构造：

    .. code-block:: python
    
        >>> import oneflow as flow
        
        >>> p = flow.placement("cuda", ranks=[0, 1, 2, 3])
        >>> p
        oneflow.placement(type="cuda", ranks=[0, 1, 2, 3])
        >>> p = flow.placement("cuda", ranks=[[0, 1], [2, 3]])
        >>> p
        oneflow.placement(type="cuda", ranks=[[0, 1], [2, 3]])

"""
)