import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.optim.lr_scheduler.ExponentialLR,
    """
    在每个 epoch 中以 gamma 衰减每个参数组的学习率。
    当 last_epoch=- 1时，设置初始学习率为 lr。

    参数：
        - **optimizer** (Optimizer) - 封装的优化器。
        - **gamma** (float) - 学习率衰减的乘法系数。
        - **last_step** (int) - 最后一个 epoch 的索引，默认值：-1。
        - **verbose** (bool) - 如果为 ``True`` ，则会为每次更新打印一条信息到 stdout，默认值: ``False`` 。

    """
)

reset_docstr(
    oneflow.optim.lr_scheduler.ExponentialLR.get_lr,
    """
    利用学习率调整器链式计算学习率。
    """
)