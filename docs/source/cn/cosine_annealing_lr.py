import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.optim.lr_scheduler.CosineAnnealingLR,
    r"""
    文档参考自： https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html?highlight=cosine#torch.optim.lr_scheduler.CosineAnnealingLR

    使用 cosine annealing schedule 设置每个参数组的学习率，其中 :math:`\eta_{max}` 被设置为初始学学习率，
    :math:`T_{cur}` 是自 SGDR 中最后一次重启以来的 epoch 数。

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    当 last_step=-1， 设置初始学习率为 lr。因为调整是递归定义的， 除学习率调整器之外，它也可以同时被其他算子修改。
    如果学习率完全由这个调整器设定，每一步的学习率就变成了：

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    它在 `SGDR: Stochastic Gradient Descent with Warm Restarts`_ 中被提出。
    请注意，这只实现了 SGDR 的 cosine annealing 部分，并不包括重新启动。

    参数:
        - **optimizer** (Optimizer) - 被包装的优化器。
        - **T_max** (int) - 迭代的最大数量。
        - **eta_min** (float) - 学习率的最小值，默认值：0。
        - **last_step** (int) - 最后一个 epoch 的索引，默认值：-1。
        - **verbose** (bool) - 如果为 ``True`` ，则会为每次更新打印一条信息到标准输出，默认值: ``False`` 。

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """
)
