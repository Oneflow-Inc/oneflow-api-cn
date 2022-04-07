import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.nn.functional.layer_norm,
    """nn.functional.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05) -> Tensor

    对最后一定数量的维度应用图层标准化。

    更多细节请参考 :class:`~oneflow.nn.LayerNorm`。

    """
)
