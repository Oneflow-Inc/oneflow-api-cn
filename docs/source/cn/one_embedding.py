import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.one_embedding.MultiTableEmbedding,
    r"""MultiTableEmbedding 表示具有相同 embedding_dim、dtype 和 key_type 的多个 embedding 表。

    参数：
        - **name** (str) - 实例化的 Embedding 层的名字。
        - **embedding_dim** (int) - 每一个 embedding 向量的大小。
        - **dtype** (flow.dtype) - embedding 的数据类型。
        - **key_type** (flow.dtype) - 特征 ID 的数据类型。
        - **tables** (list) - 由 flow.one_embedding.make_table_options 生成的列表中的表格参数。
        - **store_options** (dict) - Embedding 的存储选项。
        - **default_initializer** (dict, optional) - 如果参数 tables 为 None，则使用 default_initializer 来初始化表。默认为 None。
    
    示例：

    .. code-block:: python

        > import oneflow as flow
        > import numpy as np
        > import oneflow.nn as nn
        > # 一个使用三个表的简单样例
        > table_size_array = [39884407, 39043, 17289]
        > vocab_size = sum(table_size_array)
        > num_tables = len(table_size_array)
        > embedding_size = 128
        > scales = np.sqrt(1 / np.array(table_size_array))
        > tables = [
        >     flow.one_embedding.make_table_options(
        >         flow.one_embedding.make_uniform_initializer(low=-scale, high=scale)
        >     )
        >     for scale in scales
        > ]
        > store_options = flow.one_embedding.make_cached_ssd_store_options(
        >     cache_budget_mb=8192, persistent_path="/your_path_to_ssd", capacity=vocab_size,
        > )
        > embedding = flow.one_embedding.MultiTableEmbedding(
        >     name="my_embedding",
        >     embedding_dim=embedding_size,
        >     dtype=flow.float,
        >     key_type=flow.int64,
        >     tables=tables,
        >     store_options=store_options,
        > )
        > embedding.to("cuda")
        > mlp = flow.nn.FusedMLP(
        >     in_features=embedding_size * num_tables,
        >     hidden_features=[512, 256, 128],
        >     out_features=1,
        >     skip_final_activation=True,
        > )
        > mlp.to("cuda")
        >
        > class TrainGraph(flow.nn.Graph):
        >     def __init__(self,):
        >         super().__init__()
        >         self.embedding_lookup = embedding
        >         self.mlp = mlp
        >         self.add_optimizer(
        >             flow.optim.SGD(self.embedding_lookup.parameters(), lr=0.1, momentum=0.0)
        >         )
        >         self.add_optimizer(
        >             flow.optim.SGD(self.mlp.parameters(), lr=0.1, momentum=0.0)
        >         ) 
        >     def build(self, ids):
        >         embedding = self.embedding_lookup(ids)
        >         loss = self.mlp(flow.reshape(embedding, (-1, num_tables * embedding_size)))
        >         loss = loss.sum()
        >         loss.backward()
        >         return loss 
        > ids = np.random.randint(0, 1000, (100, num_tables), dtype=np.int64)
        > ids_tensor = flow.tensor(ids, requires_grad=False).to("cuda")
        > graph = TrainGraph()
        > loss = graph(ids_tensor)
        > print(loss)

    """
)

reset_docstr(
    oneflow.one_embedding.MultiTableEmbedding.save_snapshot,
    """保存快照。

    参数：       
        - **snapshot_name** (str) - 快照的名称。快照将被保存在路径 your_configed_persistent_path 下的 snapshots 目录中。
    
    示例：

        .. code-block:: python

            > import oneflow as flow
            > # 使用由 flow.one_embedding.MultiTableEmbedding 创建的 embedding
            > embedding.save_snapshot("my_snapshot1")
            > # 一个名为 "my_snapshot1" 的快照已经被保存在 "snapshots "目录下，可以在 
            > # your_configed_persistent_path 目录下，通过 flow.one_embedding.load_snapshot重新加载。
        """
)

reset_docstr(
    oneflow.one_embedding.MultiTableEmbedding.load_snapshot,
    """加载快照。

        参数：
            - **snapshot_name** (str) - 快照的名称。快照将从路径 your_configed_persistent_path 下加载。
    
        示例：

        .. code-block:: python

            > import oneflow as flow
            > # 使用由 flow.one_embedding.MultiTableEmbedding 创建的 embedding
            > embedding.load_snapshot("my_snapshot1")
            > # 在 your_configed_persistent_path 目录下，加载名为 "my_snapshot1" 的快照。
        """
)

reset_docstr(
    oneflow.one_embedding.MultiTableEmbedding.forward,
    """MultiTableEmbedding 的前向计算。

        参数：
            - **ids** (flow.tensor) - 特征 id。
            - **table_ids** (flow.tensor, optional) - 每个 id 的 table_id 必须与 ids 的形状相同。如果只配置了一个表或者 id 的形状为（batch_size, num_tables），并且每个列的 id 都属于表的 column_id，就无需传递 table_ids，否则，应该传递 tensor_ids。
            
        返回：
            - **flow.tensor** - embedding 查询的结果。
            
        """
)

reset_docstr(
    oneflow.one_embedding.make_device_mem_store_options,
    """将 MultiTableEmbedding 的参数 store_options 配置为（词表使用）纯 GPU 存储。

    参数：
        - **persistent_path** (str, list) - Embedding 词表的持久化存储路径。如果传入一个 str，当前 rank 中的 Embedding 词表将被存储在路径 path/rank_id-num_ranks 下；如果传入一个 list，则列表长度必须等于 rank 的数量，列表中的每个元素代表着对应 rank_id 中的 Embedding 词表的存储路径。
        - **capacity** (int) - Embedding 词表的总容量。
        - **size_factor** (int, optional) - embedding_dim 的存储大小因子。如果使用 SGD 优化器且 momentum = 0，其值应为 1；如果 momentum > 0，其值应为 2；如果使用 Adam 优化器，其值应为 3。默认值为 1。
        - **physical_block_size** (int, optional) - 是扇区大小。默认为 512。

    返回值类型：
        dict

    查看 :func:`oneflow.one_embedding.make_cached_ssd_store_options` 来获得其他信息。
    """
)

reset_docstr(
    oneflow.one_embedding.make_cached_ssd_store_options,
    """将 MultiTableEmbedding 的参数 store_options 配置为（词表使用）SSD 存储，且使用 GPU 作为高速缓存。

    参数：
        - **cache_budget_mb** (int) - 单个 GPU 显存作为缓存的预算，单位为 MB。
        - **persistent_path** (str, list) - Embedding 词表的持久化存储路径。必须使用高速 SSD 由于训练期间频繁的随机磁盘访问。如果传入一个 str，当前 rank 中的 Embedding 词表将被存储在路径 path/rank_id-num_ranks 下；如果传入一个 list，则列表长度必须等于 rank 的数量，列表中的每个元素代表着对应 rank_id 中的 Embedding 词表的存储路径。
        - **capacity** (int) - Embedding 词表的总容量
        - **size_factor** (int, optional) - embedding_dim 的存储大小因子。如果使用 SGD 优化器且 momentum = 0，其值应为 1；如果 momentum > 0，其值应为 2；如果使用 Adam 优化器，其值应为 3。默认值为 1。
        - **physical_block_size** (int, optional) - 是扇区大小。默认为 512。

    返回值类型：
        dict

    示例：

    .. code-block:: python

        >>> import oneflow as flow    
        >>> store_options = flow.one_embedding.make_cached_ssd_store_options(
        >>>     cache_budget_mb=8192, persistent_path="/your_path_to_ssd", capacity=vocab_size,
        >>> )
        >>> # pass the store_options to the "store_options" param of flow.one_embedding.MultiTableEmbedding
        >>> # ...
    """
)

reset_docstr(
    oneflow.one_embedding.make_cached_host_mem_store_options,
    """将 MultiTableEmbedding 的参数 store_options 配置为（词表使用）CPU 内存存储，且使用 GPU 作为高速缓存。

    参数：
        - **cache_budget_mb** (int) - 单个 GPU 显存作为缓存的预算，单位为 MB。
        - **persistent_path** (str, list) - Embedding 词表的持久化存储路径。如果传入一个 str，当前 rank 中的 Embedding 词表将被存储在路径 path/rank_id-num_ranks 下；如果传入一个 list，则列表长度必须等于 rank 的数量，列表中的每个元素代表着对应 rank_id 中的 Embedding 词表的存储路径。
        - **capacity** (int) - Embedding 词表的总容量。
        - **size_factor** (int, optional) - embedding_dim 的存储大小因子。如果使用 SGD 优化器且 momentum = 0，其值应为 1；如果 momentum > 0，其值应为 2；如果使用 Adam 优化器，其值应为 3。默认值为 1。
        - **physical_block_size** (int, optional) - 是扇区大小。默认为 512。

    返回值类型：
        dict

    查看 :func:`oneflow.one_embedding.make_cached_ssd_store_options` 来获得其他信息。
    """
)

reset_docstr(
    oneflow.one_embedding.make_uniform_initializer,
    """生成函数 make_table_options 所需的均匀分布初始化器参数。

    参数：
        - **low** (float) - 要生成的随机数范围的下限，应为标量。
        - **high** (float) - 要生成的随机数范围的上限，应为标量。

    返回值类型：
        dict
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> initializer = flow.one_embedding.make_uniform_initializer(low=-scale, high=scale)
        >>> # pass the initializer to flow.one_embedding.make_table_options
        >>> # ...
    """
)

reset_docstr(
    oneflow.one_embedding.make_normal_initializer,
    """生成函数 make_table_options 所需的正态分布初始化器参数。

    参数：
        - **mean** (float) - 要生成的随机数的平均值，应为标量。
        - **std** (float) - 要生成的随机数的标准差，应为标量。

    返回值类型：
        dict
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> initializer = flow.one_embedding.make_normal_initializer(mean=0, std=0.01)
        >>> # pass the initializer to flow.one_embedding.make_table_options
        >>> # ...
    """
)

reset_docstr(
    oneflow.one_embedding.make_table_options,
    """生成 MultiTableEmbedding 的参数 tables 中的元素 table。

    参数：
        - **initializer** (dict) - 初始化器，由 make_uniform_initializer 或 make_normal_initializer 生成。

    返回值类型：
        dict
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> initializer = flow.one_embedding.make_uniform_initializer(low=-scale, high=scale)
        >>> table1 = flow.one_embedding.make_table_options(initializer)
        >>> table2 = flow.one_embedding.make_table_options(initializer)
        >>> tables = [table1, table2]
        >>> # pass the tables to the "tables" param of flow.one_embedding.MultiTableEmbedding
        >>> # ...
        
    """
)

reset_docstr(
    oneflow.one_embedding.make_table,
    """`oneflow.one_embedding.make_table_options` 的别名函数。

    查看 :func:`oneflow.one_embedding.make_table_options` 来获得更多细节。
    """
)
