import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.nms,
    r"""narrow(x, dim, start, length) -> Tensor

    根据盒子的交叉联合（IoU），对盒子执行非最大抑制（NMS）。

    NMS迭代地删除那些与另一个（高分）盒子的 IoU 大于 iou_threshold 的低分盒子。

    参数：
        - **boxes** (Tensor[N, 4]) - 要执行 NMS 的盒子。它们应该是 `(x1, y1, x2, y2)` 格式，`0 <= x1 < x2` 和 `0 <= y1 < y2` 。
        - **scores** (Tensor[N]) - 每个盒子的分数
        - **iou_threshold** (float): 丢弃所有 IoU > iou_threshold 的重叠框。

    返回类型： 
        tensor: int64张量，包含被 NMS 保留的元素的索引，按照分数递减的顺序排序

    """
)
