import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.optim.RMSprop,
    """实现 RMSprop 优化算法。

    Root Mean Squared Propagation (RMSProp) 是一种未发表的、自适应的学习率方法。最初的构想提出了 RMSProp，在
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf 的第29页。

    原始方程如下：

    .. math::

        r(w, t) = \\alpha r(w, t-1) + (1 - \\alpha)(\\nabla Q_{i}(w))^2

        W = w - \\frac{\\eta} {\\\\sqrt{r(w,t) + \\epsilon}} \\nabla Q_{i}(w)

    第一个方程计算了每个权重的梯度平方的移动平均值。然后将梯度除以 :math:`sqrt{v(w,t)}`.
    在一些情况下，加入一个 momentum 项 :math: `\\beta` 是很有效的。在这个优化算法实现的过程中，使用了 Nesterov momentum:

    .. math::

        r(w, t) = \\alpha r(w, t-1) + (1 - \\alpha)(\\nabla Q_{i}(w))^2

        v(w, t) = \\beta v(w, t-1) + \\frac{\\eta} {\\\\sqrt{r(w,t) +
            \\epsilon}} \\nabla Q_{i}(w)

        w = w - v(w, t)

    如果 centered 为 True:

    .. math::

        r(w, t) = \\alpha r(w, t-1) + (1 - \\alpha)(\\nabla Q_{i}(w))^2

        g(w, t) = \\alpha g(w, t-1) + (1 - \\alpha)\\nabla Q_{i}(w)

        v(w, t) = \\beta v(w, t-1) + \\frac{\\eta} {\\\\sqrt{r(w,t) - (g(w, t))^2 +
            \\epsilon}} \\nabla Q_{i}(w)

        w = w - v(w, t)

    其中 :math:`\\alpha` 是一个超参数，其典型值为 0.99， 0.95 等等。 :math:`\\beta` 是一个 momentum 项。
    :math:`\\epsilon` 是一个 smoothing 项，以避免被零整除，通常设置的范围在 1e-4 到 1e-8。


    参数:
        - **params** (iterable) - 待优化参数构成的 iterable 或定义了参数组的 dict。
        - **lr** (float, optional) - 学习率（默认值：1e-2）。 
        - **momentum** (float, optional) - momentum 因子（默认值: 0, oneflow 现在不支持 momenmtum > 0）。
        - **alpha** (float, optional) - smoothing 常量（默认值: 0.99）。
        - **eps** (float, optional) - 添加到分母中以提高数值稳定性的项（默认值：1e-8）。
        - **centered** (bool, optional) - 如果为 ``True`` ， 计算 centered RMSProp，梯度通过对其方差的估计进行标准化处理。
        - **weight_decay** (float, optional) - 权重衰减（L2 penalty）（默认值: 0）。

    示例: 

    例 1: 

    .. code-block:: python 

        # 假设 net 是一个自定义模型。 
        rmsprop = flow.optim.RMSprop(net.parameters(), lr=1e-3)

        for epoch in range(epochs):
            # 读取数据，计算损失，等等。
            # ...
            loss.backward()
            rmsprop.step()
            rmsprop.zero_grad()

    例 2: 

    .. code-block:: python 

        # 假设 net 是一个自定义模型。 
        rmsprop = flow.optim.RMSprop(
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
            rmsprop.clip_grad()
            rmsprop.step()
            rmsprop.zero_grad()

    若要使用 clip_grad 函数，请参考这个示例。

    关于 `clip_grad_max_norm` 和 `clip_grad_norm_type` 函数的更多细节，请参考 :func:`oneflow.nn.utils.clip_grad_norm` 。

    """
)

reset_docstr(
    oneflow.optim.RMSprop.step,
    """执行一个优化步骤。

        参数:
            - **closure** (callable, optional) - 重新测试模型并返回损失的闭包。
        """
)

