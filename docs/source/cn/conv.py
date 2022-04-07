import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow._C.conv2d,
    r"""
    conv2d(input, weight, bias=None, stride=[1], padding=[0], dilation=[1], groups=1) -> Tensor

    文档引用自： https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html?highlight=conv2d

    对由多个输入平面组成的输入信号应用二维卷积。

    请参阅 :class:`~oneflow.nn.Conv2d` 获取有关详细信息和输出形状。

    参数：
        - **input** - 形状为 :math:`(\text{minibatch} , \text{in_channels} , iH , iW)` 的量化输入张量
        - **weight** - 形状为 :math:`(\text{out_channels} , \frac{\text{in_channels}}{\text{groups}} , kH , kW)` 的量化滤波器
        - **bias** - 形状为 :math:`(\text{out_channels})` 的非量化偏置张量，其类型必须为 `flow.float`。
        - **stride** - 卷积核的步长。可以是单个数字或元组 `(sH, sW)`， 默认值：1。
        - **padding** - 输入两侧的隐式填充。可以是单个数字或元组 `(padH, padW)`，默认值：0。
        - **dilation** - 内核元素之间的间距。可以是单个数字或元组 `(dH, dW)`，默认值：1。
        - **groups** - 将输入分成应该可以被组数整除的 :math:`\text{in_channels}`，默认值：1。
    
    
        """
)

reset_docstr(
    oneflow._C.conv3d,
    r"""
    conv3d(input, weight, bias=None, stride=[1], padding=[0], dilation=[1], groups=1) -> Tensor

    文档引用自： https://pytorch.org/docs/stable/generated/torch.nn.functional.conv3d.html?highlight=conv3d

    对由多个输入平面组成的输入信号应用三维卷积。

    请参阅 :class:`~oneflow.nn.Conv3d` 获取有关详细信息和输出形状。

    参数：
        - **input** - 形状的量化输入张量 :math:`(\text{minibatch} , \text{in_channels} , iD , iH , iW)`
        - **weight** - 形状的量化滤波器 :math:`(\text{out_channels} , \frac{\text{in_channels}}{\text{groups}} , kD , kH , kW)`
        - **bias** - 非量化的形状的偏置张量 :math:`(\text{out_channels})`。张量类型必须为 `flow.float`。
        - **stride** - 卷积核的步长。可以是单个数字或元组 `(sD, sH, sW)`， 默认值：1。
        - **padding** - 输入两侧的隐式填充。可以是单个数字或元组 `(padD, padH, padW)`，默认值：0。
        - **dilation** - 内核元素之间的间距。可以是单个数字或元组 `(dD, dH, dW)`，默认值：1。
        - **groups** - 将输入分成应该可以被组数整除的 :math:`\text{in_channels}`，默认值：1。
        
        
    """
)