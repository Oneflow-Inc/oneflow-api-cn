import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.utils.data.BatchSampler,
    r"""
    将另一个采样器包装为 mini-batch 索引。

    参数：
        - **sampler** (Sampler or Iterable) - 原采样器。可以是任何可迭代对象。 
        - **batch_size** (int) - mini-batch 的大小 
        - **drop_last** (bool) - 如果为 ``True`` ，采样器会在最后一个 batch 小于 ``batch_size`` 时将其采取 dropout 操作。

    样例：
        >>> import oneflow
        >>> from oneflow.utils.data import BatchSampler
        >>> from oneflow.utils.data import SequentialSampler
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """,

    

)

reset_docstr(
    oneflow.utils.data.ConcatDataset,
    r"""
    一个由多个数据集串接而成的数据集。这个类可用于聚集不同的已有数据集。

    参数：
        - **datasets** (sequence) - 被串接的数据集列表



    """
    ,
)

reset_docstr(
    oneflow.utils.data.DataLoader,

    r"""
    数据加载器。结合一个数据集和采样器，并为数据集提供一个可迭代对象。
    数据加载器同时支持 map-style 和 iterable-style 数据集的单/多线程加载，且可以自定义加载顺序和可选的自动处理和内存锁定。
    查看 :py:mod:`flow.utils.data` 文档页面以获取更多信息。

    
    参数：
        - **dataset** (Dataset) - 用于提取并加载数据的数据集
        - **batch_size** (int, 可选) - 每个 batch 加载的样本数量 (默认： ``1``)
        - **shuffle** (bool, 可选) - 设置为 ``True`` 使得数据在每个时刻重新打乱 (默认： ``False``)
        - **sampler** (Sampler or Iterable, 可选) - 定义从数据库中抽取样本的策略。可以是任何采用了 ``__len__`` 的 ``Iterable``。如果该参数已被指定，则 :attr:`shuffle` 无法被指定
        - **batch_sampler** (Sampler or Iterable, 可选) - 类似 :attr:`sampler`, 但每次返回一个 batch 索引。与 :attr:`batch_size`, :attr:`shuffle` , :attr:`sampler`,和 :attr:`drop_last` 互斥
        - **num_workers** (int, 可选) - 被用于加载数据的子线程数量 (默认： ``0``). ``0`` 意味着所有数据都将在主进程加载
        - **collate_fn** (callable, 可选) - 将一个样本列表合并成 mini-batch 张量。在从 map-style 数据集中使用批加载时被调用
        - **drop_last** (bool, 可选) - 设置为 ``True`` 以在数据集大小无法被 batch 大小整除时将最后一个不完整的 batch 采取 dropout 操作。当为 ``False`` 时，如果数据集大小无法被 batch 大小整除，最后一个 batch 将相对较小 (默认： ``False``)
        - **timeout** (numeric, 可选) - 若为正数，该参数为从 worker 处收集 batch 的 timeout 值。应总为非负值 (默认： ``0``)
        - **worker_init_fn** (callable, 可选) - 如果不是 ``None``, 该参数将在 seeding 之后和加载数据前，与 worker id (an int in ``[0, num_workers - 1]``)一起在每个 worker 的子进程被调用，以作为输入 (默认： ``None``)
        - **prefetch_factor** (int, 可选, 键word-only arg) - 在每个 worker 之前就被加载的样本数量 ``2``意味着将会有 2 * num_workers 个样本在所有 worker 之前被预提取。(default: ``2``)
        - **persistent_workers** (bool, 可选) - 如果为 ``True``, 数据加载器将不会在数据集被加载后关闭 worker 进程。这将允许 `Dataset` worker 进程保持活跃 (默认： ``False``)

    .. warning:: 如果 ``spawn`` 启动方法被使用, :attr:`worker_init_fn`
                不能为 unpicklable 对象, e.g., lambda 函数。

    .. warning:: ``len(dataloader)`` 启动法将基于采样器的长度。
                 当 :attr:`dataset` 为 :class:`~flow.utils.data.IterableDataset` 时,
                 它将返回一个基于 ``len(dataset) / batch_size`` 的估算值，并将根据 :attr:`drop_last` 进行适当取整，
                 不考虑多线程加载配置。这是 OneFlow 能做出的最理想估算，因为 OneFlow 相信 user :attr:`dataset` 代码能够正确的处理多线程加载
                 来避免重复数据。 

                 但是，如果多个 worker 的数据分片有不完全的末尾 batch ，此类估算仍有可能为不准确的，因为
                 (1) 一个原本完整的 batch 可能被分割为多个，且(2) 当 :attr:`drop_last` 被设定时，原本能合成一个以上
                 batch 的样本将被采取 dropout 操作。遗憾的是， OneFlow 通常无法检测这类情况。
    """,
)

reset_docstr(
    oneflow.utils.data.Dataset,
    r"""一个代表 :class:`Dataset` 的抽象类。

    所有能将键映射为数据样本的数据集都应作为该类的子类。所有子类都应替换 :meth:`__getitem__` 以支持使用指定键提取数据样本。子集也能可选性的替换 :meth:`__len__` ，一个在许多 :class:`~flow.utils.data.Sampler` 实现和 :class:`~flow.utils.data.DataLoader` 的默认选项中返回数据集大小的参数。

    .. note::
      :class:`~flow.utils.data.DataLoader` 在默认情况下会构建一个生成整型索引的索引采样器。
      若要处理由非整型的索引/键映射成的数据集，则需要提供一个自定义采样器。
    """,
)

reset_docstr(
    oneflow.utils.data.IterableDataset,
    r"""一个可迭代数据集。

    所有包含可迭代的数据样本的数据集都应作为此类的子类。此类数据集在数据来源为数据流时尤其有效。

    所有子类都应替换 :meth:`__iter__` ，一个用于返回本数据集中迭代器的参数。

    当一个子类和 :class:`~flow.utils.data.DataLoader` 同时使用时，数据集中的每个内容都将从 :class:`~flow.utils.data.DataLoader` 中的迭代器中生成。当 :attr:`num_workers > 0` 时，每个 worker 进程都将拥有一个数据集对象的不同拷贝，所以通常偏好将每个拷贝独立配置以避免 worker 返回重复数据。

    样例1：在 :meth:`__iter__` 中将负荷分配给所有 worker：
    
    .. code-block:: python

        >>> import oneflow as flow
        >>> class MyIterableDataset(flow.utils.data.IterableDataset):
        ...     def __init__(self, start, end):
        ...         super(MyIterableDataset).__init__()
        ...         assert end > start, "this example code only works with end >= start"
        ...         self.start = start
        ...         self.end = end
        ...
        ...     def __iter__(self):
        ...         iter_start = self.start
        ...         iter_end = self.end
        ...         return iter(range(iter_start, iter_end))
        ...
        >>> # 应提供数据为 range(3,7) 的数据集，i.e., [3, 4, 5, 6].
        >>> ds = MyIterableDataset(start=3, end=7)

        >>> # 单线程加载
        >>> print(list(flow.utils.data.DataLoader(ds, num_workers=0)))
        [tensor([3], dtype=oneflow.int64), tensor([4], dtype=oneflow.int64), tensor([5], dtype=oneflow.int64), tensor([6], dtype=oneflow.int64)]

    样例2：使用 :attr:`worker_init_fn` 将负荷分配给所有 worker：

    .. code-block:: python

        >>> import oneflow as flow
        >>> class MyIterableDataset(flow.utils.data.IterableDataset):
        ...     def __init__(self, start, end):
        ...         super(MyIterableDataset).__init__()
        ...         assert end > start, "this example code only works with end >= start"
        ...         self.start = start
        ...         self.end = end
        ...
        ...     def __iter__(self):
        ...         return iter(range(self.start, self.end))
        ...
        >>> # 应提供数据为 range(3,7) 的数据集，i.e., [3, 4, 5, 6].
        >>> ds = MyIterableDataset(start=3, end=7)

        >>> # 单线程加载
        >>> print(list(flow.utils.data.DataLoader(ds, num_workers=0)))
        [tensor([3], dtype=oneflow.int64), tensor([4], dtype=oneflow.int64), tensor([5], dtype=oneflow.int64), tensor([6], dtype=oneflow.int64)]

    """
)

reset_docstr(
    oneflow.utils.data.RandomSampler,
        
    r"""将元素随机采样。如果没有替换，则直接采样一个完整的打乱的数据集。如果有替换，则用户可以指定抽取的数量。

    参数：
        - **data_source** (Dataset) - 用于采样的数据集
        - **replacement** (bool) - 为 ``True`` 时，样本将根据指定方式被抽取，默认为 ``False`` 
        - **num_samples** (int) - 抽取的数量，默认为 `len(dataset)` 。该参数只有在 `replacement` 为 ``True`` 时需要被指定
        - **generator** (Generator) - 采样中被使用的生成器。
    """,
)

reset_docstr(
    oneflow.utils.data.Sampler,

    r"""所有采样器的基本类。

    所有采样器子类都需要提供一个用于迭代数据集元素的 :meth:`__iter__` 方法，和一个返回迭代器长度的 :meth:`__len__` 方法。

    .. note:: :meth:`__len__` 方法并不严格的在 :class:`~flow.utils.data.DataLoader` 中被要求，但是在任何有关计算 :class:`~flow.utils.data.DataLoader` 的长度的情况下都是必须的。
    """
)

reset_docstr(
    oneflow.utils.data.SequentialSampler,
    r"""依次采样元素，总是以相同的顺序。

    参数：
        - **data_source** (Dataset) - 用于采样的数据集
    """,
)

reset_docstr(
    oneflow.utils.data.Subset,
    r"""
    一个由数据集的指定索引形成的子集。

    参数：
        - **dataset** (Dataset) - 整个数据集
        - **indices** (sequence) - 被选入子集的索引
    """
)

reset_docstr(
    oneflow.utils.data.SubsetRandomSampler,
    r"""根据给定的索引列表随机采样元素，没有替换机制。

    参数：
        - **indices** (sequence) - 一个索引的序列
        - **generator** (Generator) - 用于采样的生成器
    """
)

reset_docstr(
    oneflow.utils.data.TensorDataset,
    r"""包装张量的数据集。

    每个样本都将通过索引张量的第一个维度来提取。

    参数：
        - **tensors** (tensor) - 第一维度大小相同的张量
    """
)

reset_docstr(
    oneflow.utils.data.random_split,
    """
    将数据集随机拆分为给定长度的非重叠新数据集。可选择使用生成器以获得可重复的结果。

    .. code-block:: python

        >   random_split(range(10), [3, 7], generator=flow.Generator().manual_seed(42))

    参数：
        - **dataset** (Dataset) - 待拆分的数据集
        - **lengths** (sequence) - 新数据集的长度
        - **generator** (Generator) - 用于随机排列的生成器

    """
)

reset_docstr(
    oneflow.utils.data.distributed.DistributedSampler,
    """
    将数据加载到数据集子集的采样器。

    它与 :class:`flow.nn.parallel.DistributedDataParallel` 结合使用时特别有效。在这种情况下，每个进程都可以将 :class:`~flow.utils.data.DistributedSampler` 实例作为 :class:`~flow.utils.data.DataLoader` 采样器传递，并加载它独有的原始数据集的子集。

    .. note::
        假定数据集大小不变。

    参数：
        - **dataset** - 用于采样的数据集
        - **num_replicas** (int, optional) - 参与分布式训练的进程数。默认情况下，:attr:`world_size` 从当前分布式组中检索。
        - **rank** (int, optional) - 当前进程在 :attr:`num_replicas` 中的 rank。默认情况下，:attr:`rank` 是从当前分布式组中检索的。
        - **shuffle** (bool, optional) - 如果为 ``True`` （默认值），采样器将打乱索引
        - **seed** (int, optional) - 如果 :attr:`shuffle=True`，:attr:`seed` 是用于打乱采样器的随机种子。此数字在分布式组中的所有进程中应相同。默认值为 0。
        - **drop_last** (bool, optional) - 如果为 ``True``，采样器将丢弃数据的尾部以使其在副本数量上均匀可分。如果为 ``False``，采样器将添加额外的索引，以使数据在副本中均匀可分。默认值为 ``False``。

    .. warning::
        在分布式模式下，当每个 epoch 开始时，创建 :class:`DataLoader` 迭代器 **之前** 调用 :meth:`set_epoch` 方法是必要的，
        以使 shuffle 在多个 epoch 中正常工作。否则，将始终使用相同的顺序。

    示例：

    .. code-block:: python

        >   sampler = DistributedSampler(dataset) if is_distributed else None
        >   loader = DataLoader(dataset, shuffle=(sampler is None), sampler=sampler)
        >   for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """
)

reset_docstr(
    oneflow.utils.data.distributed.DistributedSampler.set_epoch,
    """设置采样器的 epoch。
        
        当 :attr:`shuffle=True` 时，可以确保所有副本对每个 epoch 使用不同的随机排序。否则，此采样器的下一次迭代将产生相同的排序。

        参数：
            - **epoch** (int) - Epoch 数量
    """

)
