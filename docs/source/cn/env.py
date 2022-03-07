import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.env.get_world_size,
    r"""
    返回当前进程组包含的机器数量。

    返回值：
        进程组中包含的机器数量。
    
    """
)