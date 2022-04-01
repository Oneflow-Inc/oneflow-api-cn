import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.optim.SGD,
    """实现 SGD 优化算法。

    该算法将随机样本的梯度作为小批量梯度下降中整体梯度的近似估计。

    当 momentum = 0，参数的更新方程为：

        .. math::

            param_{new} = param_{old} - learning\\_rate * grad

    当有了 momentum，参数的更新方程为：

        .. math::

            & V_t = \\beta * V_{t-1} - learning\\_rate * (g_t + param_{old} * weight\\_decay)

            & param_{new} = param_{old} + V_t

    参数:
        - **params** (iterable) - 待优化参数构成的 iterable 或定义了参数组的 dict。
        - **lr** (float, optional) - 学习率（默认值：1e-3）。 
        - **momentum** (float, optional) - momentum 因子（默认值: 0.0）。
        - **weight_decay** (float, optional) - 权重衰减（L2 penalty）（默认值: 0.0）。

    示例： 

    例1:：

    .. code-block:: python 

        # 假设 net 是一个自定义模型。
        sgd = flow.optim.SGD(net.parameters(), lr=1e-3)

        for epoch in range(epochs):
            # 读取数据，计算损失，等等。
            # ...
            loss.backward()
            sgd.step()
            sgd.zero_grad()

    例2： 

    .. code-block:: python 

        # 假设 net 是一个自定义模型。
        sgd = flow.optim.SGD(
            [
                {
                    "params": net.parameters(),
                    "lr": learning_rate,
                    "clip_grad_max_norm": 0.5,
                    "clip_grad_norm_type": 2.0,
                }
            ],
        )

        for epoch in range(epochs):
            # 读取数据，计算损失，等等。
            # ...
            loss.backward()
            sgd.clip_grad()
            sgd.step()
            sgd.zero_grad()

    若要使用 clip_grad 函数，请参考这个示例。

    关于 `clip_grad_max_norm` 和 `clip_grad_norm_type` 函数的更多细节，请参考 :func:`oneflow.nn.utils.clip_grad_norm` 。

    """
)

