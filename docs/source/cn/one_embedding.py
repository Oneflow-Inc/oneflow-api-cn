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
    """Embedding lookup 操作。

        参数：
            - **ids** (flow.tensor) - 特征 id。
            - **table_ids** (flow.tensor, optional) - 每个 id 的 table_id 必须与 ids 的形状相同。如果只配置了一个表或者 id 的形状为（batch_size, num_tables），并且每个列的 id 都属于表的 column_id，就无需传递 table_ids，否则，应该传递 tensor_ids。
            
        返回：
            - **flow.tensor** - embedding 查询的结果。
            
        """
)

reset_docstr(
    oneflow.one_embedding.MultiTableMultiColumnEmbedding,
    r"""MultiTableMultiColumnEmbedding 表示多个 embedding 表，它们具有多个 embedding_dim，且 dtype、key_type 均相同。

    参数：
        - **name** (str) - 实例化的 Embedding 层的名字。
        - **embedding_dim** (list) - 每一个 embedding 向量的大小构成的列表。
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
        > # 一个使用三个表的简单例子，每个表有两列，第一列 embedding_size 是 10，第二列是 1。
        > # 每个表的第一列初始化为 uniform(-1/sqrt(table_size), 1/sqrt(table_size))，第二列初始化为 normal(0, 1/sqrt(table_size))
        > table_size_array = [39884407, 39043, 17289]
        > vocab_size = sum(table_size_array)
        > num_tables = len(table_size_array)
        > embedding_size_list = [10, 1]
        > scales = np.sqrt(1 / np.array(table_size_array))
        > tables = [
        >     flow.one_embedding.make_table_options(
        >       [flow.one_embedding.make_column_options(    
        >         flow.one_embedding.make_uniform_initializer(low=-scale, high=scale)), 
        >        flow.one_embedding.make_column_options(    
        >         flow.one_embedding.make_normal_initializer(mean=0, std=scale))]
        >     )
        >     for scale in scales
        > ]
        > store_options = flow.one_embedding.make_cached_ssd_store_options(
        >     cache_budget_mb=8192, persistent_path="/your_path_to_ssd", capacity=vocab_size,
        > )
        > embedding = flow.one_embedding.MultiTableMultiColumnEmbedding(
        >     name="my_embedding",
        >     embedding_dim=embedding_size_list,
        >     dtype=flow.float,
        >     key_type=flow.int64,
        >     tables=tables,
        >     store_options=store_options,
        > )
        > embedding.to("cuda")
        > mlp = flow.nn.FusedMLP(
        >     in_features=sum(embedding_size_list) * num_tables,
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
        >         loss = self.mlp(flow.reshape(embedding, (-1, num_tables * sum(embedding_size_list))))
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

        > import oneflow as flow    
        > store_options = flow.one_embedding.make_cached_ssd_store_options(
        >     cache_budget_mb=8192, persistent_path="/your_path_to_ssd", capacity=vocab_size,
        > )
        > # 将 store_options 传递给 flow.one_embedding.MultiTableEmbedding 中的 "store_options" 参数。
        > # ...
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

        > import oneflow as flow
        > initializer = flow.one_embedding.make_uniform_initializer(low=-scale, high=scale)
        > # 将初始化器传给 flow.one_embedding.make_table_options
        > # ...
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

        > import oneflow as flow
        > initializer = flow.one_embedding.make_normal_initializer(mean=0, std=0.01)
        > # 将初始化器传递给 flow.one_embedding.make_table_options
        > # ...
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

        > import oneflow as flow
        > initializer = flow.one_embedding.make_uniform_initializer(low=-scale, high=scale)
        > table1 = flow.one_embedding.make_table_options(initializer)
        > table2 = flow.one_embedding.make_table_options(initializer)
        > tables = [table1, table2]
        > # 将表传递给 flow.one_embedding.MultiTableEmbedding 的 ``tables`` 参数。
        > # ...
        
    """
)

reset_docstr(
    oneflow.one_embedding.make_table,
    """`oneflow.one_embedding.make_table_options` 的别名函数。

    查看 :func:`oneflow.one_embedding.make_table_options` 来获得更多细节。
    """
)

reset_docstr(
    oneflow.one_embedding.Ftrl,
    r"""FTRL 优化器。

    公式为： 

    .. math:: 

        & accumlator_{i+1} = accumlator_{i} + grad * grad
            
        & sigma = (accumulator_{i+1}^{lr\_power} - accumulator_{i}^{lr\_power}) / learning\_rate
            
        & z_{i+1} = z_{i} + grad - sigma * param_{i}

        \text{}
            param_{i+1} = \begin{cases}
        0 & \text{ if } |z_{i+1}| < \lambda_1 \\
        -(\frac{\beta+accumlator_{i+1}^{lr\_power}}{learning\_rate} + \lambda_2)*(z_{i+1} - sign(z_{i+1})*\lambda_1) & \text{ otherwise } \\
        \end{cases}
    
    示例1: 

    .. code-block:: python 

        # 假设 net 是一个自定义模型。 
        adam = flow.one_embedding.FTRL(net.parameters(), lr=1e-3)

        for epoch in range(epochs):
            # 读取数据，计算损失等。
            # ...
            loss.backward()
            adam.step()
            adam.zero_grad()

    参数：
        - **params** (Union[Iterator[Parameter], List[Dict]]) - 待优化参数构成的 iterable 或定义了参数组的 dict。
        - **lr** (float, optional) - 学习率，默认值：1e-3。
        - **weight_decay** (float, optional) - 权重衰减（L2 penalty），默认值：0.0。
        - **lr_power** (float, optional) - 学习率下降系数，默认值：-0.5。
        - **initial_accumulator_value** (float, optional) - accumlator 的初始值，默认值：0.1。
        - **lambda1** (float, optional) - L1 正则化强度，默认值：0.0。
        - **lambda2** (float, optional) - L2 正则化强度，默认值：0.0。
        - **beta** (float, optional) - beta 的值，默认值：0.0。
    """
)

reset_docstr(
    oneflow.one_embedding.Ftrl.step,
    """执行一个优化步骤。
        
        参数：
            - **closure** (callable, optional) - 重新测试模型并返回损失的闭包。
        """
)

