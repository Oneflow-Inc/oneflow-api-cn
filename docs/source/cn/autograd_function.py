import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.autograd.Function,
        r"""
    Function(self)

    创建自定义 autograd.Function 的基类。

    要创建一个自定义的 autograd.Function，需要继承这个类并实现静态函数 ``forward()`` 和 ``backward()`` 。 
    此后，若要在前向计算中使用自定义算子，调用子类中的函数 ``apply()`` 或者 ``__call__()`` ，不要直接调用 ``forward()`` 函数。

    示例：

    .. code-block:: python

        class Exp(Function):
            @staticmethod
            def forward(ctx, i):
                result = i.exp()
                ctx.save_for_backward(result)
                return result

            @staticmethod
            def backward(ctx, grad_output):
                result, = ctx.saved_tensors
                return grad_output * result

        # 通过调用 apply 函数或 __call__ 函数来使用
        output = Exp.apply(input)  # output = Exp()(input)
    """
)

reset_docstr(
    oneflow.autograd.Function.__call__,
    r"""
        参考 :meth:`self.apply` 。
        """
)

reset_docstr(
    oneflow.autograd.Function.apply,
    r"""
        计算输出张量并构建反向传播计算图。
        """
)

