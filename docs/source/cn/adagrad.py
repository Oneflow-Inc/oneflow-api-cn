import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.optim.Adagrad,
    r"""实现 Adagrad 优化算法。 

        公式是: 

        .. math:: 

            & S_{t} = S_{t-1} + grad \odot grad 
            
            & decay\_lr = \frac{learning\_rate}{(1 + (train\_step - 1) * lr\_decay)}

            & X_{t} = X_{t-1} - \frac{decay\_lr}{\sqrt{S_{t} + \epsilon}} \odot grad

        参数:
            - **params** (Union[Iterator[Parameter], List[Dict]])- 待优化参数构成的 iterable 或定义了参数的 dict。
            - **lr** (float, optional)- 学习率，默认为 0.001。
            - **lr_decay** (float, optional)- 学习率的衰减因子，默认为 0.0。
            - **weight_decay** (float, optional)- 权重衰减， 默认为 0。
            - **initial_accumulator_value** (float, optional)- S 的初始值，默认为 0.0。
            - **eps** (float, optional)- 一个为提高数值稳定性而添加到分母的小常数项，默认为 1e-10。
        
        例如: 

        例1: 

        .. code-block:: python

            # 假设 net 是一个自定义模型。
            adagrad = flow.optim.Adagrad(net.parameters(), lr=1e-3)

            for epoch in range(epochs):
                # Read data, Compute the loss and so on. 
                # ...
                loss.backward()
                adagrad.step()
                adagrad.zero_grad()

        例2: 

        .. code-block:: python 

            # 假设 net 是一个自定义模型。
            adagrad = flow.optim.Adagrad(
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
                # 读取数据，计算损失等等。
                # ...
                loss.backward()
                adagrad.clip_grad()
                adagrad.step()
                adagrad.zero_grad()

        如果你想要使用 clip_grad 函数，你可以参考这个例子。

        关于 `clip_grad_max_norm` 和 `clip_grad_norm_type` 的更多细节, 你可以参考 :func:`oneflow.nn.utils.clip_grad_norm_` 。 
        
        """
)

reset_docstr(
    oneflow.optim.Adagrad.step,
    r"""执行单个优化步骤。

        参数:
            - **closure** (callable, optional)- 重新测试模型并返回损失的闭包。          
    """
)
