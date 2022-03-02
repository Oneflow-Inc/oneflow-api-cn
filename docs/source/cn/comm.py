import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.comm.send,
    r"""send(input, dst, send_meta=True)

    同步发送张量。  

    参数：
        - **input** (tensor): 要发送的张量
        - **dst** (int): 目的地rank
        - **send_meta** (bool): 是否发送元信息（默认为 True）

    """,    
)

reset_docstr(
    oneflow.comm.recv,
    r"""recv(src, shape=None, dtype=None, device=None, *, out=None)
    
    同步接收张量。
    
    如果 send_meta = False，所有 `shape` 和 `dtype` 要有值，否则全部为 None 。

    参数：
        - **src** (int, optional): 来源 rank 。如果未指定，将从任何进程接收。
        - **shape** (optional): 输出张量形状。
        - **dataType** (optional): 输出张量数据类型。
        - **device** (optional): 输出张量设备。
        - **out** (Tensor, optional): 填充接收到的数据的张量。
    
    返回类型：
        如果 `out` 为 None，则返回接收到的张量。否则 out 将获取数据而不返回数据。

    """,
)

reset_docstr(
    oneflow.comm.all_gather,
    r"""
    将整个进程组的张量收集到一个列表中。

    参数：
        - tensor_list (list[Tensor]): 输出列表。它应该包含收集到的正确尺寸的张量用于输出。
        - tensor (Tensor): 从当前进程广播的张量。

    样例：

    .. code-block:: python

        >>> # 我们有一个进程组，包含两个 rank。
        >>> import oneflow as flow

        >>> input = flow.tensor([[1, 2], [3, 4]], device="cuda") + flow.env.get_local_rank()
        >>> # rank0 的输入
        >>> input # doctest: +ONLY_CHECK_RANK_0
        tensor([[1, 2],
                [3, 4]], device='cuda:0', dtype=oneflow.int64)
        >>> # rank1 的输入
        >>> input # doctest: +ONLY_CHECK_RANK_1
        tensor([[2, 3],
                [4, 5]], device='cuda:1', dtype=oneflow.int64)
        >>> tensor_list = [flow.zeros(2, 2, dtype=flow.int64) for _ in range(2)]
        >>> flow.comm.all_gather(tensor_list, input)
        >>> # rank0 的结果
        >>> tensor_list # doctest: +ONLY_CHECK_RANK_0
        [tensor([[1, 2],
                [3, 4]], device='cuda:0', dtype=oneflow.int64), tensor([[2, 3],
                [4, 5]], device='cuda:0', dtype=oneflow.int64)]
        >>> # rank1 的结果
        >>> tensor_list # doctest: +ONLY_CHECK_RANK_1
        [tensor([[1, 2],
                [3, 4]], device='cuda:1', dtype=oneflow.int64), tensor([[2, 3],
                [4, 5]], device='cuda:1', dtype=oneflow.int64)]

    """,
)