import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.autograd.backward,
    r"""
    文档参考自：https://pytorch.org/docs/stable/generated/torch.autograd.backward.html#torch.autograd.backward 。

    计算给定张量相对于图中叶子结点的梯度之和。

    使用链式法则对该 graph 进行微分。如果 ``tensors`` 中不全是标量（即它的某个数据有一个以上的元素）且需要梯度，
    那么将计算雅各比向量积，此时，该函数还需要指定 ``grad_tensors``。这应当是一个长度匹配的序列，
    包含通常是对应张量的微分函数的梯度的位于雅各布向量积中的“向量”。（若所有张量都不需要梯度，``None`` 也是可接受的值）

    此算子累计了叶子结点的梯度——你可能需要将 ``.grad`` 属性归零或者在调用前设置它们为 ``None``。

    Note:
        设置 ``create_graph=True`` 来调用这个函数会造成一个参数和其梯度之间的循环引用，这可能会导致内存泄漏。当创建 graph 时，
        我们推荐使用 ``autograd.grad`` 来避免这个问题。如果你必须使用这个函数，确保重置参数中的 ``.grad`` 字段为 ``None`` 
        以打破循环，避免内存泄漏。

    参数：
        - **tensors** (Tensor or Sequence[Tensor]) - 将被计算导数的张量。
        - **grad_tensors** (Tensor or Sequence[Tensor], optional) - 雅各比向量乘积中的“向量”，通常对相应张量的每个元素进行梯度计算。(对于标量张量或不需要梯度的张量，可以指定为 None)
        - **retain_graph** (bool, optional) - 如果为 ``False`` ，用来计算梯度的计算图将会在反向传播完成后被重置。默认值： ``False`` 。注意在几乎所有的情况下都不需要把这个选项设置为 ``True``，通常这可以用更有效的方法来解决。默认为 ``create_graph`` 的值。
        - **create_graph** (bool, optional) - 如果为 ``True`` ， 将构建导数的计算图，允许计算高阶导数，默认为 ``False``。
    """
)

reset_docstr(
    oneflow.autograd.grad,
    r"""
    文档参考自： https://pytorch.org/docs/stable/generated/torch.autograd.grad.html#torch.autograd.grad 。

    计算并返回输出相对于输入的梯度之和。

    使用链式法则对该计算图进行微分。``grad_outputs`` 应该是一个长度与 ``outputs`` 相匹配的序列，包含雅各比向量积中的“向量”。
    （ ``None`` 是一个可接受的值，该张量不需要梯度。）

    参数：
        - **outputs** (Sequence[Tensor]) - 将被计算导数的张量。
        - **inputs** (Sequence[Tensor]) - 导数将被返回的输入张量（而不是累积到 ``.grad``）。
        - **grad_outputs** (Sequence[Tensor], optional) - 雅各比向量乘积中的“向量”，通常是相对于每个输出的梯度。对于标量张量或者不需要梯度的张量，可以指定值为 None。默认值为 None。
        - **retain_graph** (bool, optional) - 如果为 ``False`` ，用来计算梯度的 graph 将会在后向计算完成后被重置。默认值： ``False`` 。注意在几乎所有的情况下都不需要把这个选项设置为 ``True`` ，通常可以用更有效的方法来解决。默认为 ``create_graph`` 的值。
        - **create_graph** (bool, optional) - 如果为 ``True`` ， 将构建导数的 graph，允许计算高阶导数结果，默认为 ``False`` 。

    返回值：
        - **Tuple** (Tensor) - 包含每个 ``input`` 梯度的张量元组。
    """
)