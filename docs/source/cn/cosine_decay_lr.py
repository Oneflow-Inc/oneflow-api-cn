import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.optim.lr_scheduler.CosineDecayLR,
    """这个算子构建了一个 Cosine decayed 学习率调整器。

    在用户指定 decay_steps 之前，学习率将被更新为：

    .. math::

        & cos\\_decay = 0.5*(1+cos(\\pi*\\frac{current\\_step}{decay\\_steps}))

        & decay\\_factor = (1-\\alpha)*cos\\_decay+\\alpha

        & learning\\_rate = base\\_learning\\_rate*decay\\_factor

    在用户指定 decay_steps 之后，学习率将是：

    .. math::

        learning\\_rate = {base\\_learning\\_rate}*{\\alpha}

    它在 `SGDR: Stochastic Gradient Descent with Warm Restarts`_ 中被提出。
    请注意，这里只实现了 SGDR 的 cosine annealing 部分，而不是重新启动。

    参数：
        - **optimizer** (Optimizer) - 被包装的优化器。
        - **decay_steps** (int) - 学习率调整器中 decay steps。
        - **alpha** (float, optional) - 学习率比例因子（ :math:`\\alpha` ）（默认值： 0.0）。
        - **last_step** (int) - 最后一个 epoch 的索引，默认值：-1。
        - **verbose** (bool) - 如果为 ``True`` ，则会为每次更新打印一条信息到标准输出，默认值: ``False`` 。

    示例：

    .. code-block:: python

        import oneflow as flow

        ...
        cosine_decay_lr = flow.optim.lr_scheduler.CosineDecayLR(optimizer, decay_steps=100, alpha=0.0)
        for epoch in range(num_epoch):
            train(...)
            cosine_decay_lr.step()

    .. _SGDR\\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """
)


