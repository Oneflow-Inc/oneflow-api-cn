import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.optim.lr_scheduler.LambdaLR,
    """
    将每个参数组的学习率设置为初始 lr 乘一个给定的函数。
    当 last_epoch=- 1时，设置初始学习率为 lr。

    .. math::

        learning\\_rate = base\\_learning\\_rate*lambda(last\\_step)

    参数：
        - **optimizer** (Optimizer) - 封装的优化器。
        - **lr_lambda** (function or list) - 一个给定整数参数 epoch 的计算乘法因子的函数，
            或者一个此类函数的列表，一个在每组 optimizer.param_groups 中。
        - **last_step** (int) - 最后一个 epoch 的索引（默认值：-1）。
        - **verbose** (bool) - 如果为 ``True`` ，则会为每次更新打印一条信息到 stdout（默认值: ``False`` ）。

    示例：

    .. code-block:: python

        import oneflow as flow

        ...
        lambda1 = lambda step: step // 30
        lambda2 = lambda step: 0.95 * step
        lambda_lr = flow.optim.lr_scheduler.LambdaLR(optimizer, [lambda1, lambda2])
        for epoch in range(num_epoch):
            train(...)
            lambda_lr.step()

    """
)