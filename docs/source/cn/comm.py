import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.comm.send,
    r"""send(input, dst, send_meta=True)

    同步地发送一个张量。  

    参数：
        - **input** (tensor) - 要发送的张量
        - **dst** (int) - 目标 rank
        - **send_meta** (bool) - 是否发送元信息（默认值为 True）

    """,
)

reset_docstr(
    oneflow.comm.recv,
    r"""recv(src, shape=None, dtype=None, device=None, *, out=None)
    
    同步地接收一个张量。
    
    如果 send_meta = False，那么 `shape`、`dtype` 和 `device` 都应有值，否则应全为 None。

    参数：
        - **src** (int, optional) - 源数据的 rank。如果未指定，将从任何进程接收。
        - **shape** (optional) - 输出张量的形状。
        - **dataType** (optional) - 输出张量数据的类型。
        - **device** (optional) - 输出张量的设备。
        - **out** (Tensor, optional) - 使用接收数据填充的张量。
    
    返回值：
        如果 `out` 为 `None`，则返回接收到的张量。否则将从 `out` 自身获取数据而不返回。

    """,
)

reset_docstr(
    oneflow.comm.all_reduce,
    """
    对所有机器上的张量做规约操作，结果返回给所有进程。

    参数：
        - **tensor** (Tensor) - 输入张量

    示例：

    .. code-block:: python

        > # We have 1 process groups, 2 ranks.
        > import oneflow as flow
        > tensor = flow.tensor([[1, 2], [3, 4]], device="cuda") + flow.env.get_local_rank()
        > # tensor on rank0
        > tensor
        tensor([[1, 2],
                [3, 4]], device='cuda:0', dtype=oneflow.int64)
        > # tensor on rank1
        > tensor
        tensor([[2, 3],
                [4, 5]], device='cuda:1', dtype=oneflow.int64)
        > flow.comm.all_reduce(tensor)
        > tensor.numpy()
        array([[3, 5],
               [7, 9]], dtype=int64)

    """
)

reset_docstr(
    oneflow.comm.reduce_scatter,
    r"""
    对输入列表中的张量做规约操作，然后将其分发到一个进程组中的所有进程。

    参数：
        - **output** (Tensor) - 输出张量
        - **input_list** (list[Tensor]) - 被规约并分发的张量的列表
    """
)

reset_docstr(
    oneflow.comm.scatter,
    r"""
    将输入列表中的张量分发到一个进程组中的所有进程。

    每个进程都会准确地接收到一个张量，并将其数据存储在 ``tensor`` 参数中。


    参数：
        - **tensor** (Tensor) - 输出张量
        - **scatter_list** (list[Tensor]) - 被分发的张量列表（默认值为 None，在源 rank 上必须被指定）
        - **src** (int) - 源数据的 rank (默认值为 0)
    """
)

reset_docstr(
    oneflow.comm.reduce,
    r"""
    对所有机器上的张量做规约操作。

    只有 rank 为 ``dst`` 的进程可以接收到结果。

    参数：
        - **tensor** (Tensor) - 输入和输出的张量集合。函数以 in-place 方式操作。
        - **dst** (int) - 目标 rank。

    """

)

reset_docstr(
    oneflow.comm.broadcast,
    r"""
    将张量广播到进程组中的所有进程中。``tensor`` 在所有参与广播的进程中必须拥有相同的元素数量。

    参数：
        - **tensor** (Tensor) - 如果 ``src`` 为当前进程的 rank 则为被发送的张量，否则为用于保存接收数据的张量。
        - **src** (int) - 源数据的 rank。

    .. code-block:: python

        >   # We have 1 process groups, 2 ranks.
        >   import oneflow as flow
        >   tensor = flow.tensor([[1, 2], [3, 4]], device="cuda") + flow.env.get_local_rank()
        >   # input on rank0
        >   tensor
        tensor([[1, 2],
                [3, 4]], device='cuda:0', dtype=oneflow.int64)
        >   # input on rank1
        >   tensor
        tensor([[2, 3],
                [4, 5]], device='cuda:1', dtype=oneflow.int64)
        >   flow.comm.broadcast(tensor, 0)
        >   # result on rank0
        >   tensor
        tensor([[1, 2],
                [3, 4]], device='cuda:0', dtype=oneflow.int64)

    """
)

reset_docstr(
    oneflow.comm.gather,
    r"""
    将单个进程上的张量收集到一个列表中。

    参数：
        - **tensor** (Tensor) - 输入张量
        - **gather_list** (list[Tensor], 可选) - 用于收集数据的适当大小的张量列表（默认值为 None，在源 rank 上必须被指定）
        - **dst** (int, 可选) - 目标 rank (默认值为 0)

    """
)

reset_docstr(
    oneflow.comm.all_gather,
    r"""all_gather(tensor_list, tensor)
    
    将整个进程组的张量收集到一个列表中。

    参数：

        - **tensor_list** (list[Tensor]) - 输出列表。它应该包含正确大小的张量，用于整体的输出。
        - **tensor** (Tensor) - 从当前进程广播的张量。

    示例：


    .. code-block:: python

        > # We have 1 process groups, 2 ranks.
        > import oneflow as flow
        > input = flow.tensor([[1, 2], [3, 4]], device="cuda") + flow.env.get_local_rank()
        > # input on rank0
        > input
        tensor([[1, 2],
                [3, 4]], device='cuda:0', dtype=oneflow.int64)
        > # input on rank1
        > input
        tensor([[2, 3],
                [4, 5]], device='cuda:1', dtype=oneflow.int64)
        > tensor_list = [flow.zeros(2, 2, dtype=flow.int64) for _ in range(2)]
        > flow.comm.all_gather(tensor_list, input)
        > # result on rank0
        > tensor_list
        [tensor([[1, 2],
                [3, 4]], device='cuda:0', dtype=oneflow.int64), tensor([[2, 3],
                [4, 5]], device='cuda:0', dtype=oneflow.int64)]
        > # result on rank1
        > tensor_list
        [tensor([[1, 2],
                [3, 4]], device='cuda:1', dtype=oneflow.int64), tensor([[2, 3],
                [4, 5]], device='cuda:1', dtype=oneflow.int64)]
    """,
)
