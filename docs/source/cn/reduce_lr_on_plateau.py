import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.optim.lr_scheduler.ReduceLROnPlateau,
    """当一个 metric 停止更新时，降低学习率。
    一旦学习停滞不前，模型通常会从降低2-10倍的学习率中受益。
    这个调整器读取一个指标量，如果在 'patience' 的 epoch 数中没有看到改进，
    学习率就会降低。

    参数:
        - **optimizer** (Optimizer) - 封装的优化器。
        - **mode** (str) - `min` 或者 `max` 。在 `min` 模式中，当监测的数量不再减少时学习率将会减小；
            在 `max` 模式中，当监测的数量不再增加时学习率将会减小。默认： `min` 。
        - **factor** (float) - 学习率降低的系数， new_lr = lr * factor ， 默认值：0.1。
        - **patience** (int) - 不会再改善模型的 epoch 数，在这之后学习率将降低。
            例如，如果 `patience = 2` ，那么我们会忽略前两个没有改善的 epoch ，
            如果 loss 仍然没有减小，学习率也将会在第三个 epoch 后减小，
            默认值：10。
        - **threshold** (float) - 测量新的最佳状态的阈值，只关注一些重要变化。默认：1e-4。
        - **threshold_mode** (str) - `rel` 或 `abs` 。 在 `rel` 模式中，
            'max' ：dynamic_threshold = best * ( 1 + threshold )，
            'min' ：dynamic_threshold = best * ( 1 - threshold )； 
            在 `abs` 模式中，`max` ：dynamic_threshold = best + threshold，
            `min` ：dynamic_threshold = best - threshold inmode。默认： `rel` 。
        - **cooldown** (int) - 学习率减少后，在恢复正常操作前要等待的 epoch 数，默认值：0。
        - **min_lr** (float or list) - 一个标量或一个标量列表，
            分别是所有参数组或每组的学习率下限，默认：0。
        - **eps** (float) - 适用于学习率的最小衰减，如果新旧学习率之间的差异小于 eps，更新将被忽略。默认值：1e-8。
        - **verbose** (bool) - 如果为 ``True`` ，则会为每次更新打印一条信息到 stdout，默认值: ``False`` 。

    示例:
    
    .. code-block:: python

        optimizer = flow.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = flow.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        for epoch in range(10):
            train(...)
            val_loss = validate(...)
            # 注意，该步骤应在validate()之后调用。
            scheduler.step(val_loss)
    """
)

reset_docstr(
    oneflow.optim.lr_scheduler.ReduceLROnPlateau.load_state_dict,
    """加载调整器的状态。

        参数：
            - **state_dict** (dict) - 调整器的状态，应为调用 :meth:`state_dict` 函数所返回的对象。
        """
)

reset_docstr(
    oneflow.optim.lr_scheduler.ReduceLROnPlateau.state_dict,
    """ 以 :class:`dict` 形式返回调整器的状态。

        它包含了 self.__dict__ 中每个变量的条目，而这些变量并不是优化器。
        """
)

reset_docstr(
    oneflow.optim.lr_scheduler.ReduceLROnPlateau.step,
    """执行单个优化步骤。

        参数:
            - **metrics** (float)- 一个衡量模型训练效果的指标量。

        """
)

