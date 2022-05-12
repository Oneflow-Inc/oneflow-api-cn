import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.get_rng_state,
    r"""该文档引用自：
    https://pytorch.org/docs/1.10/generated/torch.get_rng_state.html

    设置随机数生成器状态。

    参数:
        - **new_state** (oneflow.ByteTensor) - 所需的状态

    """
)

reset_docstr(
    oneflow.initial_seed,
    r"""该文档引用自：
    https://pytorch.org/docs/1.10/_modules/torch/random.html.

    返回用于生成随机数的初始种子，作为一个 Python `long`。

    """
)

reset_docstr(
    oneflow.manual_seed,
    r"""该文档引用自：
    https://pytorch.org/docs/1.10/generated/torch.manual_seed.html

    设置生成随机数的种子。返回一个 `oneflow.Generator` 对象。

    参数:
        - **seed** (int) - 所需的 seed。该值必须在以下范围内 ` [ -0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff ] ` 。否则，会产生一个 RuntimeError，负的输入被重新映射为正值，公式为 ` 0xffff_ffff_ffff_ffff + seed ` 。

    """
)

reset_docstr(
    oneflow.seed,
    r"""该文档引用自：
    https://pytorch.org/docs/1.10/generated/torch.seed.html.

    将生成随机数的种子设置为一个非决定性的随机数。返回一个用于 RNG 种子的 64 位数字。

    """
)

reset_docstr(
    oneflow.set_rng_state,
    r"""该文档引用自：
    https://pytorch.org/docs/1.10/generated/torch.set_rng_state.html.

    返回随机数发生器的状态为 `oneflow.ByteTensor`。

    """
)