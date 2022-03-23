import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.nn.Module.add_module,
    """将子模块添加到当前模块。

    可以使用给定名称作为属性访问模块。

    参数：
        - **name** (string) - 子模块的名称。可以使用给定名称从此模块访问子模块。
        - **module** (Module) - 要添加到模块的子模块。

    返回值：
        Module: self
    """
)

reset_docstr(
    oneflow.nn.Module.cpu,
    """将所有模型参数和缓存移动到 CPU。

    .. note::
        此方法会以 in-place 方式修改模块。

    """
)

reset_docstr(
    oneflow.nn.Module.cuda,
    r"""将所有模型参数和缓存移动到 GPU。

    这也会使关联的参数和缓存成为不同的对象。因此，在构建优化器之前，如果模块在进行优化时驻留在 GPU 上，则应该调用它。

    .. note::
        此方法会以 in-place 方式修改模块。

    参数：
        - **device** (int, optional) - 如果指定，所有参数将被复制到该设备

    返回值：
        Module: self
    """
)

reset_docstr(
    oneflow.nn.Module.double,
    r"""将所有浮点参数和缓存转换为 ``double`` 数据类型。

    .. note::
        此方法会以 in-place 方式修改模块。

    返回值：
        Module: self
    """
)

reset_docstr(
    oneflow.nn.Module.extra_repr,
    """设置此模块的其余表现形式。

    要打印自定义的额外信息，应在模块中重新实现此方法。接受单行和多行字符串。
    """
)

reset_docstr(
    oneflow.nn.Module.float,
    r"""将所有浮点参数和缓存转换为 ``float`` 数据类型。

    .. note::
        此方法会以 in-place 方式修改模块。

    返回值：
        Module: self
    """
)

# reset_docstr(
#     oneflow.nn.Module.to_consistent,
#     """
#     此接口已停止维护，请使用 :func:`oneflow.nn.Module.to_global` 
#     """
# )

# reset_docstr(
#     oneflow.nn.Module.to_global,
#     """
#     将所有参数和缓存设置为全局的。

#     它对此模块中每一个参数和缓存执行相同的 :func:`oneflow.Tensor.to_global` 

#     Note:
#         此方法会以 in-place 方式修改模块。

#         如果该模块的参数和缓存是本地的，则 placement 和 sbp 都是必需的，否则至少需要其中的一个。

#     参数：
#         - **placement** (flow.placement, optional) - 该模块中参数和缓存的所需位置。默认值为 None
#         - **sbp** (flow.sbp.sbp or tuple of flow.sbp.sbp, optional) - 此模块中参数和缓存的所需 sbp。默认值为 None

#     示例：

#     .. code-block:: python

#         >   import oneflow as flow
#         >   m = flow.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3)
#         >   m.to_global(placement=flow.placement("cpu", ranks=[0]), sbp=[flow.sbp.split(0)])
#         >   m.weight.is_global
#         True
#         >   m.bias.is_global
#         True
#     """
# )

reset_docstr(
    oneflow.nn.Module.zero_grad,
    r"""将所有模型参数的梯度设置为零。更多细节请参考 :class:`oneflow.optim.Optimizer` 中的类似功能。

    参数：
        - **set_to_none** (bool) - 设置梯度为 None 而非 0。更多细节请参考 :meth:`oneflow.optim.Optimizer.zero_grad` 

    """
)
