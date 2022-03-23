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
    oneflow.comm.all_reduce,
    r"""
    对所有机器上的张量做规约操作，结果返回给所有进程。

    参数：
        - **tensor** (Tensor): 输入张量

    示例：

    .. code-block:: python

        # 我们有一个进程组，两个 rank 。
        > import oneflow as flow

        > tensor = flow.tensor([[1, 2], [3, 4]], device="cuda") + flow.env.get_local_rank()
        # rank0 上的 tensor
        > tensor
        tensor([[1, 2],
                [3, 4]], device='cuda:0', dtype=oneflow.int64)

        # rank1 上的 tensor
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
        - **output** (Tensor): 输出张量
        - **input_list** (list[Tensor]): reduce 并分散的张量的列表
    """
)

reset_docstr(
    oneflow.comm.scatter,
    r"""
    将一列张量分发到一个进程组中的所有进程，

    每个进程将接收到正好一个张量，并将数据储存在 ``tensor`` 参数中。


    参数：
        - **tensor** (Tensor): 输出张量
        - **scatter_list** (list[Tensor]): 被分散的张量列表（默认为 None ，需要指定源 rank）
        - **src** (int): 源数据的 rank (默认为0)
    """
)

reset_docstr(
    oneflow.comm.reduce,
    r"""
    对所有机器的张量数据做规约操作。

    只有拥有 rank ``dst`` 的进程将接收到 reduce 后的结果。

    参数：
        - **tensor** (Tensor): 输入和输出的张量集合。参数使用 in-place 操作。
        - **dst** (int): 终点 rank

    """

)

reset_docstr(
    oneflow.comm.broadcast,
    r"""
    将张量广播到所有进程组中。
    ``tensor`` 在位于组内的所有进程中必须拥有相同的元素数量。

    参数：
        - **tensor** (Tensor): 如果 ``src`` 为当前张量的 rank ，则为被发送的张量，否则为接收的张量。
        - **src** (int): 源数据的 rank

    .. code-block:: python

        > # 我们有一个进程组，两个 rank 。
        > import oneflow as flow
        > tensor = flow.tensor([[1, 2], [3, 4]], device="cuda") + flow.env.get_local_rank()
        > # rank0 的输入
        > tensor 
        tensor([[1, 2],
                [3, 4]], device='cuda:0', dtype=oneflow.int64)
        > # rank1 的输入
        > tensor 
        tensor([[2, 3],
                [4, 5]], device='cuda:1', dtype=oneflow.int64)
        > flow.comm.broadcast(tensor, 0)
        > # rank0 的结果
        > tensor 
        tensor([[1, 2],
                [3, 4]], device='cuda:0', dtype=oneflow.int64)

    """
)

reset_docstr(
    oneflow.comm.gather,
    r"""
    将单个线程上的一列张量收集。

    参数：
        - **tensor** (Tensor): 输入张量
        - **gather_list** (list[Tensor], 可选): 一列大小适当的张量，用于储存聚集后的张量（默认为 None ，需要指定终点 rank ）
        - **dst** (int, 可选): 终点 rank (默认为 0)

    """
)

reset_docstr(
    oneflow.comm.all_gather,
    r"""
    all_gather(tensor_list, tensor)
    
    将整个进程组的张量收集到一个列表中。
    
    参数：
    
        - **tensor_list** (list[Tensor]) - 输出列表。它应该包含正确大小的张量，用于集体的输出。
        - **tensor** (Tensor) - 从当前进程广播的张量。
        
    样例：
    
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
