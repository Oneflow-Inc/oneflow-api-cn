import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.einsum,
    r"""
    使用基于爱因斯坦求和惯例的符号，沿指定维度对输入 :attr:`operands` 的元素的乘积进行求和。

    通过基于爱因斯坦求和惯例的简短格式，详见 :attr:`equation` ，einsum 允许计算许多常见的多维线性代数数组操作。 :attr:`equation` 格式的细节将在下面描述，但通常是用一些下标来标记输入 :attr:`operands` 的每个维度，并定义哪些下标是输出的一部分。然后沿着元素下标不属于输出的一部分的维度，对 :attr:`operands` 元素的乘积进行计算。例如，矩阵乘法可以用 einsum 计算，如 `flow.einsum("ij,jk->ik", A, B)`。这里 j 是求和的下标，i 和 k 是输出的下标（详见下面的章节）。

    Equation:
        :attr:`equation` 字符串指定了每个输入的 :attr:`operands` 维度的标号（字母 `[a-zA-Z]`），顺序与维度相同，每个 :attr:`operands` 的标号之间用逗号（','）隔开。
        例如: ``'ij,jk'`` 指定两个二维 :attr:`operands` 的标号。用同一标号标注的维度的尺寸必须是 `broadcastable`，也就是说，它们的尺寸必须匹配或为 1。例外的情况是，如果一个标号在同一个输入 :attr:`operands` 上重复出现，这个 :attr:`operands` 用这个下标标注的尺寸必须匹配，并且该 :attr:`operands` 将被其沿这个尺寸的对角线所取代。
        在 :attr:`equation` 中出现一次的标号将是输出的一部分，按字母顺序递增排序。
        输出是通过将输入的 :attr:`operands` 元素相乘来计算的，然后将子标不属于输出的尺寸相加。
        也可以通过在 :attr:`equation` 末尾添加一个箭头（'->'）来明确定义输出下标号。
        箭头右边是输出的子标。
        例如，下面的方程计算了一个矩阵乘法的转置。
        矩阵乘法：'ij,jk->ki'。对于某些输入 :attr:`operands` ，输出的标号必须至少出现一次，而对于输出，则最多出现一次。

        省略号（'...'）可以用来代替标号，表示省略的尺寸。
        每个输入 :attr:`operands` 最多可以包含一个省略号，该省略号将覆盖未被标号覆盖的尺寸。
        例如，对于一个有5个维度的输入 :attr:`operands` ，公式"'ab...c'"中的省略号覆盖第三和第四维度。
        省略号不需要覆盖整个 :attr:`operands` 的相同数量的维度，但是省略号的'形状'（它们所覆盖的维度的大小）必须一起播出。如果输出没有
        用箭头（'->'）符号明确定义，省略号将在输出中排在第一位（最左边的尺寸）。
        在输入操作数精确出现一次的下标标签之前。 例如，下面的公式实现了
        批量矩阵乘法"'...ij,...jk'"。

        最后几点说明：:attr:`equation` 中的不同元素之间可以包含空白（下标、省略号。
        箭头和逗号），但像`'. .''是无效的。空字符串"''"对标量操作数有效。

    .. note::

        `flow.einsum` 处理省略号('...')的方式与 NumPy 不同，它允许省略号覆盖的维度被求和。
        省略号所覆盖的维度进行求和，也就是说，省略号不需要成为输出的一部分。

    .. note::

        这个函数没有对给定的表达式进行优化，所以相同计算的不同公式可能会
        运行得更快或消耗更少的内存。使用 opt_einsum（https://optimized-einsum.readthedocs.io/en/stable/）可以优化公式。

    参数:
        - **equation** (String) - 爱因斯坦求和法的标号。
        - **operands** (oneflow.Tensor) - 计算爱因斯坦求和的张量。
        

    示例:

    .. code-block:: python

        >>> import oneflow as flow

        # trace
        >>> flow.einsum('ii', flow.arange(4*4).reshape(4,4).to(flow.float32))
        tensor(30., dtype=oneflow.float32)

        # diagonal
        >>> flow.einsum('ii->i', flow.arange(4*4).reshape(4,4).to(flow.float32))
        tensor([ 0.,  5., 10., 15.], dtype=oneflow.float32)

        # outer product
        >>> x = flow.arange(5).to(flow.float32)
        >>> y = flow.arange(4).to(flow.float32)
        >>> flow.einsum('i,j->ij', x, y)
        tensor([[ 0.,  0.,  0.,  0.],
                [ 0.,  1.,  2.,  3.],
                [ 0.,  2.,  4.,  6.],
                [ 0.,  3.,  6.,  9.],
                [ 0.,  4.,  8., 12.]], dtype=oneflow.float32)
        
        # batch matrix multiplication
        >>> As = flow.arange(3*2*5).reshape(3,2,5).to(flow.float32)
        >>> Bs = flow.arange(3*5*4).reshape(3,5,4).to(flow.float32)
        >>> flow.einsum('bij,bjk->bik', As, Bs).shape
        oneflow.Size([3, 2, 4])

        # batch permute
        >>> A = flow.randn(2, 3, 4, 5)
        >>> flow.einsum('...ij->...ji', A).shape
        oneflow.Size([2, 3, 5, 4])

        # bilinear
        >>> A = flow.randn(3,5,4)
        >>> l = flow.randn(2,5)
        >>> r = flow.randn(2,4)
        >>> flow.einsum('bn,anm,bm->ba', l, A, r).shape
        oneflow.Size([2, 3])

    """
)