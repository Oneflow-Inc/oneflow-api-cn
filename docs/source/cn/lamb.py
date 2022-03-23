import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.optim.LAMB,
    """实现 LAMB 优化算法。

    LAMB 在 `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_ 中被提出。

    参数更新的方程是：

    .. math::

        & V_t = \\beta_1*V_{t-1} + (1-\\beta_1)*grad

        & S_t = \\beta_2*S_{t-1} + (1-\\beta_2)*{grad} \\odot {grad}

        & \\hat{u} = \\frac{{V_t}}{\\sqrt{{S_t}}+\\epsilon}
        
        & \\hat{r} = learning\\_rate * \\frac{||param_{old}||_2}{||\\hat{u}||_2}

        & param_{new} = param_{old} - \\hat{r} * \\hat{u}

    参数:
        - **params** (iterable) - 待优化参数构成的 iterable 或定义了参数组的 dict。
        - **lr** (float, optional) - 学习率（默认值：1e-3）。 
        - **betas** (Tuple[float, float], optional) - 用于计算梯度及其平方的移动平均的系数（默认值：(0.9, 0.999))
        - **eps** (float, optional) - 添加到分母中以提高数值稳定性的项（默认值：1e-8）。
        - **weight_decay** (float, optional) - 权重衰减 (L2 penalty) (默认值: 0)
        - **adam_w_mode** (bool, optional) - 应用 L2 正则化或去耦权重衰减 True (也被称为 AdamW 优化算法) (默认值: True)
        - **do_bias_correction** (bool, optional) - 是否做偏差校正（默认值：True）。
        - **amsgrad** (bool, optional) - 是否使用该算法的 AMSGrad 变体，现在不支持! (默认值: False) 
        
    .. _Large Batch Optimization for Deep Learning\\: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962

    示例:

    例1:

    .. code-block:: python

        # 假设 net 是一个自定义模型。
        lamb = flow.optim.LAMB(net.parameters(), lr=1e-3)

        for epoch in range(epochs):
            # 读取数据，计算损失，等等。
            # ...
            loss.backward()
            lamb.step()
            lamb.zero_grad()

    Example 2:

    .. code-block:: python

        # 假设 net 是一个自定义模型。
        lamb = flow.optim.LAMB(
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
            lamb.clip_grad()
            lamb.step()
            lamb.zero_grad()

    若要使用 clip_grad 函数，请参考这个示例。

    关于 `clip_grad_max_norm` 和 `clip_grad_norm_type` 函数的更多细节，请参考 :func:`oneflow.nn.utils.clip_grad_norm` 。
    """
)

reset_docstr(
    oneflow.optim.LAMB.step,
    """执行一个优化步骤。

        参数:
            - **closure** (callable, optional) - 重新测试模型并返回损失的闭包。
        """
)

