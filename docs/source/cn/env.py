import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.env.get_node_size,
    r"""
    返回当前进程组包含的机器数量。

    返回值：
        进程组中包含的机器数量。
    
    """
)

reset_docstr(
    oneflow.env.get_local_rank,
    r"""返回当前机器的本地 rank 。本地 rank 在全局上不是独一的，其仅对于一个机器的单个进程是独一的。

    返回值：
        当前机器进程的本地 rank 。

    """
)

reset_docstr(
    oneflow.env.get_world_size,
    r"""
    返回当前进程组包含的进程数量。

    返回值：
        进程组中包含的进程数量。
    
    """
)

reset_docstr(
    oneflow.env.is_multi_client,
    r"""
    返回当前是否处于多用户模式。

    返回值：
        如果当前处于多用户模式则为真，否则为假。
    
    """
)
