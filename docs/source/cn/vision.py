import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow._C.pad,
    r"""
    填充张量。

    填充大小：
        对输入张量某些维度的填充大小的描述从最后一个维度开始，然后是前一个维度，依此类推。``input`` 的维度将被填充为  :math:`\left\lfloor\frac{\text{len(pad)}}{2}\right\rfloor` 。
        例如，若只对输入张量的最后一个维度进行填充，那么 :attr:`pad` 的参数应为 :math:`(\text{padding_left}, \text{padding_right})` ；
        若填充输入张量的最后两个维度，则参数变成 :math:`(\text{padding_left}, \text{padding_right},` :math:`\text{padding_top}, \text{padding_bottom})`；
        填充最后 3 个维度时，参数将为 :math:`(\text{padding_left}, \text{padding_right},` :math:`\text{padding_top}, \text{padding_bottom}` :math:`\text{padding_front}, \text{padding_back})`。

    填充模式：
        参考 :class:`oneflow.nn.ConstantPad2d`， :class:`oneflow.nn.ReflectionPad2d`，以及 :class:`oneflow.nn.ReplicationPad2d` 
        可以得到每个填充模式是如何工作的具体例子。对任意的尺寸实现恒定的填充。复制填充的引入是为了填充 5D 输入张量的最后 3 维，
        或 4D 输入张量的最后 2 维，或 3D 输入张量的最后一维。反射填充仅用于填充 4D 输入张量的最后两个维度，或 3D 输入张量的最后一个维度。

    参数：
        - **input** (Tensor)- N 维张量。
        - **pad** (tuple)- m 个元素的元组，其中 :math:`\frac{m}{2} \leq` 输入维度以及 :math:`m` 为偶数。
        - **mode** - ``'constant'``， ``'reflect'``， ``'replicate'`` 或者 ``'circular'``。默认值： ``'constant'``。
        - **value** - 填充值为 ``'constant'`` 的填充。默认值： ``0``。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> pad = [2, 2, 1, 1]
        >>> input = flow.tensor(np.arange(18).reshape((1, 2, 3, 3)).astype(np.float32))
        >>> output = flow.nn.functional.pad(input, pad, mode = "replicate")
        >>> output.shape
        oneflow.Size([1, 2, 5, 7])
        >>> output
        tensor([[[[ 0.,  0.,  0.,  1.,  2.,  2.,  2.],
                  [ 0.,  0.,  0.,  1.,  2.,  2.,  2.],
                  [ 3.,  3.,  3.,  4.,  5.,  5.,  5.],
                  [ 6.,  6.,  6.,  7.,  8.,  8.,  8.],
                  [ 6.,  6.,  6.,  7.,  8.,  8.,  8.]],
        <BLANKLINE>
                 [[ 9.,  9.,  9., 10., 11., 11., 11.],
                  [ 9.,  9.,  9., 10., 11., 11., 11.],
                  [12., 12., 12., 13., 14., 14., 14.],
                  [15., 15., 15., 16., 17., 17., 17.],
                  [15., 15., 15., 16., 17., 17., 17.]]]], dtype=oneflow.float32)

    参考 :class:`oneflow.nn.ConstantPad2d`, :class:`oneflow.nn.ReflectionPad2d` 和 :class:`oneflow.nn.ReplicationPad2d` ，可以得到每个填充模式是如何工作的具体例子。
        
    """
)