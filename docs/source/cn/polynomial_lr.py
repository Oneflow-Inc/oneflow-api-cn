import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.optim.lr_scheduler.PolynomialLR,
    r"""
    这个算子创建了一个多项式衰减学习率的调整器。学习率将被更新如下：

    如果 cycle 为 `True` ，则等式为：

    .. math::
        \begin{aligned}
           & decay\_batch = decay\_batch*ceil(\frac{current\_batch}{decay\_batch}) \\
           & learning\_rate = (base\_lr-end\_lr)*(1-\frac{current\_batch}{decay\_batch})^{power}+end\_lr
        \end{aligned}

    如果 cycle 为 `False` ，则等式为：

    .. math::
        \begin{aligned}
           & decay\_batch = min(decay\_batch, current\_batch) \\
           & learning\_rate = (base\_lr-end\_lr)*(1-\frac{current\_batch}{decay\_batch})^{power}+end\_lr
        \end{aligned}

    参数:
        - **optimizer** (Optimizer) - 被包装的优化器。
        - **steps** (int) - decayed steps。
        - **end_learning_rate** (float, optional) - 最终学习率，默认值为 0.0001。
        - **power** (float, optional) - 多项式的幂，默认为 1.0。
        - **cycle** (bool, optional) - 如果 cycle 为 True，调整器将在每一个 decay step 中衰减学习率，默认值为 False。
   
    示例：

    .. code-block:: python

        import oneflow as flow
       
        ... 
        polynomial_scheduler = flow.optim.lr_scheduler.PolynomialLR(
            optimizer, steps=5, end_learning_rate=0.00001, power=2
            )
        for epoch in range(num_epoch):
            train(...)
            polynomial_scheduler.step()

    """
)

