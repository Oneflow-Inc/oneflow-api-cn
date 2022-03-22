import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.nn.init.xavier_uniform_,
    r"""
    此接口与 PyTorch 一致。文档可以参考：
    https://pytorch.org/docs/stable/nn.init.html

    根据 `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010) 中描述的方法，使用均匀分布填充输入张量 `Tensor` 的值。结果张量将通过 :math:`\mathcal{U}(-a, a)` 采样得到，其中

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan_in} + \text{fan_out}}}

    此算法也被称为 Glorot initialization 。

    参数：
        - **tensor**: 一个 n 维的 `flow.Tensor`
        - **gain**: 一个可选的缩放参数

    示例：

    .. code-block:: python

        > w = flow.empty(3, 5)
        > nn.init.xavier_uniform_(w, gain=1.0)
    """
)

reset_docstr(
    oneflow.nn.init.xavier_normal_,
    r"""
    此接口与 PyTorch 一致。文档可以参考：
    https://pytorch.org/docs/stable/nn.init.html

    根据 `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010) 中描述的方法，使用正态分布填充输入张量 `Tensor` 的值。结果张量将通过 :math:`\mathcal{N}(0, \text{std}^2)` 采样得到，其中

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan_in} + \text{fan_out}}}

    此算法也被称为 Glorot initialization 。

    参数：
        - **tensor**: 一个 n 维的 `flow.Tensor`
        - **gain**: 一个可选的缩放参数

    示例：
    
    .. code-block:: python

        > w = flow.empty(3, 5)
        > nn.init.xavier_normal_(w)
    """
)

reset_docstr(
    oneflow.nn.init.kaiming_uniform_,
    r"""
    此接口与 PyTorch 一致。文档可以参考：
    https://pytorch.org/docs/stable/nn.init.html

    根据 `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015) 中描述的方法，使用均匀分布填充输入张量 `Tensor` 的值。结果张量将通过 :math:`\mathcal{U}(-\text{bound}, \text{bound})` 采样得到，其中

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan_mode}}}

    此算法也被称为 He initialization 。

    参数：
        - **tensor**: 一个 n 维的 `flow.Tensor`
        - **a**: 此层之后的激活层使用的负梯度（只能用于 ``'leaky_relu'`` ）
        - **mode**: ``'fan_in'`` (默认) 或 ``'fan_out'`` 中的一个。选择 ``'fan_in'`` 将保留正向过程中的的权重方差。选择 ``'fan_out'`` 将保留反向过程中的权重方差
        - **nonlinearity**: 非线性函数 (`nn.functional` 的名称),
            建议仅使用 ``'relu'`` 或者 ``'leaky_relu'`` (默认)

    示例：

    .. code-block:: python
    
        > w = flow.empty(3, 5)
        > nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    """

)

reset_docstr(
    oneflow.nn.init.kaiming_normal_,
    r"""
    此接口与 PyTorch 一致。文档可以参考：
    https://pytorch.org/docs/stable/nn.init.html

    根据 `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015) 中描述的方法，使用正态分布填充输入张量 `Tensor` 的值。结果张量将通过 :math:`\mathcal{N}(0, \text{std}^2)` 采样得到，其中

    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan_mode}}}

    此算法也被称为 He initialization 。

    参数：
        - **tensor**: 一个 n 维的 `flow.Tensor`
        - **a**: 此层之后的激活层使用的负梯度（只能用于 ``'leaky_relu'`` ）
        - **mode**: ``'fan_in'`` (默认) 或 ``'fan_out'`` 中的一个。选择 ``'fan_in'`` 将保留正向过程中的的权重方差。选择 ``'fan_out'`` 将保留反向过程中的权重方差
        - **nonlinearity**: 非线性函数 (`nn.functional` 的名称),
            建议仅使用 ``'relu'``  或者 ``'leaky_relu'`` (默认)

    示例：
    
    .. code-block:: python

        > w = flow.empty(3, 5)
        > nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    """

)

