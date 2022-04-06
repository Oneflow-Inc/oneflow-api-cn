import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.nn.utils.remove_weight_norm,
    r"""remove_weight_norm(module, name='weight') -> T_module

    从模块中删除 Weight Normalization Reparameterization。

    参数：
        - **module** (Module): 包含模块
        - **name** (str, 可选的): 权重参数名称

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> m = flow.nn.utils.weight_norm(flow.nn.Linear(20, 40))
        >>> flow.nn.utils.remove_weight_norm(m)
        Linear(in_features=20, out_features=40, bias=True)

    """
)

reset_docstr(
    oneflow.nn.utils.weight_norm,
    r"""
    对给定模块中的参数应用权重归一化 (Weight Normalization)。

    .. math::
        \mathbf{w}=g \frac{\mathbf{v}}{\|\mathbf{v}\|}

    权重归一化 (Weight Normalization) 是一种重新参数化 (Reparameterization)，
    它将权重张量的大小从其方向解耦。此操作将用两个参数替换由 :attr:`name` 指定的参数：
    一个指定大小（例如 ``'weight'``），另一个指定方向（例如 ``'weight_v'``)。
    权重归一化 (Weight Normalization) 是通过一个 hook 实现的，该 hook 在每个
    :meth:`~Module.forward` 调用之前从大小和方向重新计算权重张量。

    默认情况下，当 ``dim=0`` 时，每个输出通道/平面独立计算范数。
    要计算整个权重张量的范数，请使用 ``dim=None``。

    参见 https://arxiv.org/abs/1602.07868 。本文档说明参考 Pytorch 文档：https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html 。

    参数：
        - **module** (Module): 包含模块
        - **name** (str, 可选的): 权重参数名称
        - **dim** (int, 可选的): 计算范数的维度

    返回类型：
        带有权重范数 hook 的原始模块

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> m = flow.nn.utils.weight_norm(flow.nn.Linear(20, 40), name='weight')
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_g.size()
        oneflow.Size([40, 1])
        >>> m.weight_v.size()
        oneflow.Size([40, 20])

    """
)

reset_docstr(
    oneflow.nn.utils.clip_grad_norm_,
    r"""clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=True) → oneflow._oneflow_internal.Tensor

    裁剪可迭代参数的梯度范数。范数是在所有梯度上一起计算的，就像它们被连接成一个向量一样。

    参数：
        - **parameters** (Iterable[Tensor] 或 Tensor) - 一个可迭代的张量或一个将梯度归一化的单个张量。
        - **max_norm** (float 或 int) - 梯度的最大范数。
        - **norm_type** (float 或 int) - 使用的 p-norm 的类型。对于无穷范数，可以是  ``'inf'``。
        - **error_if_nonfinite** (bool) - 当为 True 时，如果来自 :attr:``parameters`` 的梯度的总范数为 ``nan`` 、 ``inf`` 或 ``-inf`` 会出现 error 。默认：True。

    返回类型：
        裁剪梯度范数后的参数。参数的总范数（视为单个向量）。

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> x1 = flow.tensor([[2, 3, 4], [1.5, 2.6, 3.7]], dtype=flow.float32, requires_grad=True)
        >>> m1 = flow.nn.ReLU()
        >>> out1 = m1(x1)
        >>> out1 = out1.sum()
        >>> out1.backward()
        >>> norm1 = flow.nn.utils.clip_grad_norm_(x1, 0.6, 1.0)
        >>> norm1
        tensor(6., dtype=oneflow.float32)
        >>> x1.grad
        tensor([[0.1000, 0.1000, 0.1000],
                [0.1000, 0.1000, 0.1000]], dtype=oneflow.float32)
        >>> x2 = flow.tensor([[-2, -3, -4], [2.5, 0, 3.2]], dtype=flow.float32, requires_grad=True)
        >>> out2 = flow.atan(x2)
        >>> out2 = out2.sum()
        >>> out2.backward()
        >>> norm2 = flow.nn.utils.clip_grad_norm_(x2, 0.5)
        >>> norm2
        tensor(1.0394, dtype=oneflow.float32)
        >>> x2.grad
        tensor([[0.0962, 0.0481, 0.0283],
                [0.0663, 0.4810, 0.0428]], dtype=oneflow.float32)

    """
)
