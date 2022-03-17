import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.optim.AdamW,
    """实现 AdamW 优化算法。

    原始的的 Adam 算法是在 `Adam: A Method for Stochastic Optimization`_ 中被提出。
    AdamW 的变式在 `Decoupled Weight Decay Regularization`_ 中被提出。

    Adam-weight-decay 算法的优化器。

    (更多详情参考 `Adam-weight-decay <https://www.fast.ai/2018/07/02/adam-weight-decay/>`_ )。

    所以我们使用 Adam-weight-decay 算法来解决以下问题。

    参数更新的方程是。

    .. math::

        & V_t = \\beta_1*V_{t-1} + (1-\\beta_1)*grad

        & S_t = \\beta_2*S_{t-1} + (1-\\beta_2)*{grad} \\odot {grad}

        & \\hat{g} = learning\\_rate*(\\frac{{V_t}}{\\sqrt{{S_t}}+\\epsilon}+\\lambda*param_{old})

        & param_{new} = param_{old} - \\hat{g}

    参数:
        - **params** (iterable) - 待优化参数构成的 iterable 或定义了参数组的 dict。
        - **lr** (float, optional) - 学习率（默认值：1e-3）。 
        - **betas** (Tuple[float, float], optional) - 用于计算梯度及其平方的运行平均数的系数（默认值：(0.9, 0.999))
        - **eps** (float, optional) - 添加到分母中以提高数值稳定性的项（默认值：1e-8）。
        - **weight_decay** (float, optional) - 权重衰减 (L2 penalty) (在等式中为 λ, 默认值: 0)
        - **amsgrad** (bool, optional) - 是否使用该算法的 AMSGrad 变体(默认值: False) 。
        - **do_bias_correction** (bool, optional) - 是否做偏差校正（默认值：True）。

    .. _Adam\\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980

    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101

    示例: 

    例1: 

    .. code-block:: python 

        # 假设 net 是一个自定义模型。
        adamw = flow.optim.AdamW(net.parameters(), lr=1e-3)

        for epoch in range(epochs):
            # 读取数据，计算损失，等等。
            # ...
            loss.backward()
            adamw.step()
            adamw.zero_grad()

    Example 2: 

    .. code-block:: python 

        # 假设 net 是一个自定义模型。
        adamw = flow.optim.AdamW(
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
            adamw.clip_grad()
            adamw.step()
            adamw.zero_grad()

    如果你想使用 clip_grad 函数，你可以参考这个示例。

    关于 `clip_grad_max_norm` 和 `clip_grad_norm_type` 函数的更多细节，你可以参考 :func:`oneflow.nn.utils.clip_grad_norm` 。

    """
)