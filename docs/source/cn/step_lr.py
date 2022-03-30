import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.optim.lr_scheduler.StepLR,
    """
    在每一个周期为 step_size 的步骤中，对每个参数组的学习率进行伽玛衰减操作。
    注意，这种衰减可以与其他因素导致的学习率变化同时发生。
    当 last_step = -1 时，设置初始学习率为 lr。

    参数:
        - **optimizer** (Optimizer) - 被包装的优化器。
        - **step_size** (int) - 学习率衰减的周期。
        - **gamma** (float, optional) - 学习率衰减的乘法系数（默认：0.1）。
        - **last_step** (int) - 最后一个 epoch 的索引（默认值：-1）。
        - **verbose** (bool) - 如果为 ``True`` ，则会在每次更新时打印一条信息到标准输出。默认为 ``False``。

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
