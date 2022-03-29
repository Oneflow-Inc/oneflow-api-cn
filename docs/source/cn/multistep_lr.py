import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.optim.lr_scheduler.MultiStepLR,
    """
    一旦步数达到一个临界值，每个参数组的学习率就会以 gamma 衰减。请注意，这种衰减可能与来自本调整器之外的其他学习率变化同时发生。
    当 last_epoch =- 1时，设置初始学习率为 lr。

    参数：
        - **optimizer** (Optimizer) - 被包装的优化器。
        - **milestones** (list) - step indices 的 list，必须是增加的。
        - **gamma** (float, optional) - 学习率衰减的乘法系数（默认值：0.1）。
        - **last_step** (int) - 最后一个 epoch 的索引（默认值：-1）。
        - **verbose** (bool) - 如果为 ``True`` ，则会为每次更新打印一条信息到标准输出（默认值: ``False`` ）。

    示例：

    .. code-block:: python

        import oneflow as flow

        ...
        multistep_lr = flow.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        for epoch in range(num_epoch):
            train(...)
            multistep_lr.step()

    """
)


