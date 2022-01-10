import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.utils.data.BatchSampler,
    r"""
    将另一个采样器包装为mini-batch索引。

    参数：
        - **sampler** (Sampler or Iterable): 原采样器。可以是任何可迭代对象。 
        - **batch_size** (int): mini-batch的大小 
        - **drop_last** (bool): 如果为 ``True`` ，采样器会在最后一个batch小于 ``batch_size`` 时将其丢弃。
    示例：


        list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """,

)

reset_docstr(
    oneflow.utils.data.ConcatDataset,
    r"""
    一个由多个数据集串接而成的数据集。这个类可用于聚集不同的已有数据集。

    参数：
        - **datasets** (sequence): 被串接的数据集列表



    """
    ,
)

reset_docstr(
    oneflow.utils.data.DataLoader,

    r"""
    数据加载器。结合一个数据集和采样器，并为数据集提供一个可迭代对象。
    数据加载器同时支持map-style和iterable-style数据集的单/多线程加载，且可以自定义加载顺序和可选的自动处理和内存锁定。
    查看 :py:mod:`flow.utils.data` 文档页面以获取更多信息。

    
    参数：
        - **dataset** (Dataset): 用于提取并加载数据的数据集
        - **batch_size** (int, 可选): 每个batch加载的样本数量 (默认： ``1``)
        - **shuffle** (bool, 可选): 设置为 ``True`` 使得数据在每个时刻Reshuffle (默认： ``False``)
        - **sampler** (Sampler or Iterable, 可选): 定义从数据库中抽取样本的策略。可以是任何采用了 ``__len__`` 的 ``Iterable``。如果该参数已被指定，则 :attr:`shuffle` 无法被指定
        - **batch_sampler** (Sampler or Iterable, 可选): 类似 :attr:`sampler`, 但每次返回一个batch索引。与 :attr:`batch_size`, :attr:`shuffle` , :attr:`sampler`,和 :attr:`drop_last` 互斥
        - **num_workers** (int, 可选): 被用于加载数据的子线程数量 (默认： ``0``). ``0`` 意味着所有数据都将在主进程加载
        - **collate_fn** (callable, 可选): 将一个样本列表合并成mini-batch张量。在从map-style数据集中使用批加载时被调用
        - **drop_last** (bool, 可选): 设置为 ``True`` 以在数据集大小无法被batch大小整除时将最后一个不完整batch丢弃。当为 ``False`` 时，如果数据集大小无法被batch大小整除，最后一个batch将相对较小 (默认： ``False``)
        - **timeout** (numeric, 可选): 若为正数，该参数为从worker处收集batch的timeout值。应总为非负值 (默认： ``0``)
        - **worker_init_fn** (callable, 可选): 如果不是 ``None``, 该参数将在seeding之后和加载数据前，与worker id (an int in ``[0, num_workers - 1]``)一起在每个worker的子进程被调用，以作为输入 (默认： ``None``)
        - **prefetch_factor** (int, 可选, keyword-only arg): 在每个worker之前就被加载的样本数量 ``2``意味着将会有 2 * num_workers 个样本在所有worker之前被预提取。(default: ``2``)
        - **persistent_workers** (bool, 可选): 如果为 ``True``, 数据加载器将不会在数据集被加载后关闭worker进程。这将允许`Dataset`worker进程保持活跃 (默认： ``False``)

    .. warning:: 如果 ``spawn`` 启动方法被使用, :attr:`worker_init_fn`
                不能为unpicklable对象, e.g., lambda函数。

    .. warning:: ``len(dataloader)`` 启动法将基于采样器的长度。
                 当 :attr:`dataset` 为 :class:`~flow.utils.data.IterableDataset` 时,
                 它将返回一个基于 ``len(dataset) / batch_size`` 的估算值，并将根据 :attr:`drop_last` 进行适当取整，
                 不考虑多线程加载配置。it instead returns an estimate based on ``len(dataset) / batch_size``, with proper
                 rounding depending on :attr:`drop_last`, regardless of multi-process loading
                 configurations.这是OneFlow能做出的最理想估算，因为OneFlow相信user :attr:`dataset` 代码能够正确的处理多线程加载
                 来避免重复数据。 

                 但是，如果多个worker的数据分片有不完全的末尾batch，此类估算仍有可能为不准确的，因为
                 (1) 一个原本完整的batch可能被分割为多个，且(2) 当 :attr:`drop_last` 被设定时，原本能合成一个以上
                 batch的样本将被丢弃。遗憾的是，OneFlow通常无法检测这类情况。
    """,
)


