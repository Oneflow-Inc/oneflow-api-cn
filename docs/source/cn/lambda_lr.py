import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.optim.lr_scheduler.LambdaLR,
    """
    将每个参数组的学习率设置为给定函数的初始 lr 倍。
    当 last_epoch=- 1时，设置初始学习率为 lr。

    .. math::

        learning\\_rate = base\\_learning\\_rate*lambda(last\\_step)

    参数：
        - **optimizer** (Optimizer) - 被包装的优化器。
        - **lr_lambda** (function or list) - 一个根据给定参数 epoch 计算乘法因子的函数，或为一组函数，其中每个函数用于 optimizer.param_groups 中的每一个组。
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

reset_docstr(
    oneflow.optim.lr_scheduler.LambdaLR.load_state_dict,
    """加载调整器的状态。

        参数：
            - **state_dict** (dict) - 调整器的状态，应为调用 :meth:`state_dict` 函数所返回的对象。
        """
)

reset_docstr(
    oneflow.optim.lr_scheduler.LambdaLR.state_dict,
    """ 以 :class:`dict` 形式返回调整器的状态。

        它包含了 self.__dict__ 中每个不是优化器的变量的条目。
        学习率 lambda 函数只有在它们是可调用的对象时才会被保存，如果它们是函数或 lambdas 则不会被保存。
        """
)