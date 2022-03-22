import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.optim.optimizer.clip_grad,
    r"""剪辑一组迭代参数的梯度范数。
        范数是在所有梯度上一起计算的，就像它们被串联成一个向量一样。
        你可以设置 max_norm 和 norm_type 。

        更多的细节可以参考文档中的优化器（如 Adam、SGD 等）。

        你也可以参考代码中的 :func:`oneflow.nn.utils.clip_grad_norm_` 函数。

    """
)

reset_docstr(
    oneflow.optim.optimizer.load_state_dict,
    r"""
        加载由 `state_dict` 函数创建的优化器的状态。

        参考自: https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.load_state_dict
        """
)

reset_docstr(
    oneflow.optim.optimizer.state_dict,
    r"""
        返回优化器的状态为 :class:`dict` .

        它包含两个参数:

        * state - 一个保存当前优化状态的 dict，它的内容在不同的优化器类之间是不同的。
        * param_group - 一个包含所有参数组的 dict。

        参考自: https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.state_dict
        """
)

reset_docstr(
    oneflow.optim.optimizer.zero_grad,
    """将所有优化的 Torch.Tensor s 的梯度设置为零。

        参数:
            - **set_to_none** (bool) - 设置 grads 为 None 而不是0。
                一般来说，这将有较低的内存占用，并能适度地提高性能。然而，会有一些不同的表现。
        例如:
            1. 当用户试图访问一个 gradient 并对其进行手动操作时，一个 None 属性或一个全是 0 的 Tensor 将表现得不同。When the user tries to access a gradient and perform manual ops on

            2. 如果用户请求 zero_grad (set_to_none=True)，然后再向后传递，对于没有收到梯度的参数，保证梯度为 None。

            3. 如果梯度为 0 或 None，优化器有不同的行为（在一种情况下，它以 0 梯度执行该步骤，而在另一种情况下，它完全跳过该步骤）

        """
)

