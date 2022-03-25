import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.optim.lr_scheduler.StepLR,
    """
    在每一个 step_size 大小的步骤中，以伽玛衰减每个参数组的学习率。
    注意，这种衰减可以与本调整器以外的其他学习率变化同时发生。
    当 last_step = -1 时，设置初始学习率为 lr。

    参数:
        - **optimizer** (Optimizer) - 封装的优化器。
        - **step_size** (int) - 学习率衰减的时间。
        - **gamma** (float, optional) - 学习率衰减的乘法系数（默认：0.1）。
        - **last_step** (int) - 最后一个 epoch 的索引（默认值：-1）。
        - **verbose** (bool) - 如果为 ``True`` ，则会为每次更新打印一条信息到 stdout（默认值: ``False`` ）。

    示例：

    .. code-block:: python

        import oneflow as flow

        ...
        step_lr = flow.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        for epoch in range(num_epoch):
            train(...)
            step_lr.step()

    """
)

reset_docstr(
    oneflow.optim.lr_scheduler.StepLR.get_lr,
    """
    利用学习率调整器链式计算学习率。
    """
)
