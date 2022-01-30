import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.nn.Graph,
    r"""在 Graph 模块下训练和评估一个神经网络的基类。

    要在 OneFlow 中使用 Graph 模块进行模型训练或评估，你应该：

    1. 将你的自定义 Graph 定义为 ``nn.Graph`` 的子类。
    2. 在子类 ``__init__()`` 中添加 ``super().__init__()`` 。
    3. 将模块作为常规属性添加到你的Graph中。
    4. 在 ``build()`` 函数中定义计算逻辑。
    5. 将 Graph 实例化并调用它。

    .. code-block:: python

        >>> import oneflow as flow

        >>> class LinearGraph(flow.nn.Graph):
        ...    def __init__(self):
        ...        super().__init__()
        ...        # 向 Graph 中添加一个模块。
        ...        self.linear = flow.nn.Linear(3, 8, False)
        ...    def build(self, x):
        ...        # 使用该模块构建 Graph 的计算逻辑。
        ...        return self.linear(x)

        # 将 Graph 实例化。
        >>> linear_graph = LinearGraph()
        >>> x = flow.randn(4, 3)

        # 第一次调用 Graph 将运行 graphs build() 方法来跟踪计算图。 
        # 计算图将首次执行并被优化。
        >>> linear_graph(x).shape
        oneflow.Size([4, 8])

        # 然后调用 Graph 将直接运行计算图。
        >>> linear_graph(x).shape
        oneflow.Size([4, 8])

    请注意 Graph 目前不可以进行嵌套。
    """
)

reset_docstr(
    oneflow.nn.Graph.__init__,
    r"""
        初始化内部 Graph 状态。 它必须在子类的 ``__init__`` 中调用。

        .. code-block:: python

            >>> import oneflow as flow
            >>> class SubclassGraph(flow.nn.Graph):
            ...     def __init__(self):
            ...         super().__init__() # 必须定义此函数
            ...         # 然后定义 Graph 的属性
            ...     def build(self):
            ...         pass

        """
)

reset_docstr(
    oneflow.nn.Graph.build,
    r""" 必须重写 ``build()`` 来定义神经网络计算逻辑。

        nn.Graph 中的 ``build()`` 与 nn.Module 中的 ``forward()`` 非常相似。
        它是用来描述计算逻辑的一个神经网络。

        当第一次调用 Graph 对象时，会隐式调用 ``build()`` 函数来构建计算图。

        确保在第一次调用 Graph 之前先调用模块中的 ``train()`` 和 ``eval()`` 函数，
        以使模块在需要时执行正确的训练或评估逻辑。

        .. code-block:: python

            >>> import oneflow as flow
            >>> class MyGraph(flow.nn.Graph):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.linear = flow.nn.Linear(3, 8, False)
            ...     def build(self, x):
            ...         return self.linear(x)

            >>> linear_graph = MyGraph()
            >>> x = flow.randn(4, 3)
            >>> y = linear_graph(x) # 隐式调用 build() 函数

        请注意， ``build()`` 函数的输入和输出目前只接受位置参数，每个参数必须是以下
        类型之一：

        * ``Tensor``
        * ``Tensor`` 中的 ``list`` 
        * ``None``

        """
)

reset_docstr(
    oneflow.nn.Graph.add_optimizer,
    r"""向 Graph 中添加一个优化器，一个学习率调整器。

        要使用 nn.Graph 进行训练，你应该再做 2 件事：

        1. 使用 ``add_optimizer()`` 函数添加至少一个优化器（学习率调整器是可选的）。
        2. 在 ``build()`` 函数中调用 loss tensor 的 ``backward()`` 函数。

        请注意，计算图将自动执行这些方法： 

        * 如果设置为梯度裁剪，则调用优化器的 ``clip_grad()`` 函数。
        * 优化器的 ``step()`` 函数。
        * 优化器的 ``zero_grad()`` 函数。
        * 学习率调整器 ``step()`` 函数。

        另请注意，暂时只允许标量张量在 ``nn.Graph.build()``  中调用 ``backward()`` 。 
        所以你可以调用 ``Tensor.sum()`` 或 ``Tensor.mean()`` 将损失张量变为标量张量。

        .. code-block:: python

            >>> import oneflow as flow
            >>> loss_fn = flow.nn.MSELoss(reduction="sum")
            >>> model = flow.nn.Sequential(flow.nn.Linear(3, 1), flow.nn.Flatten(0, 1))
            >>> optimizer = flow.optim.SGD(model.parameters(), lr=1e-6)
            >>> class LinearTrainGraph(flow.nn.Graph):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.model = model
            ...         self.loss_fn = loss_fn
            ...         # 添加一个优化器。
            ...         self.add_optimizer(optimizer)
            ...     def build(self, x, y):
            ...         y_pred = self.model(x)
            ...         loss = self.loss_fn(y_pred, y)
            ...         # 调用 loss tensor 中的 backward(), loss tensor 必须是一个标量张量。
            ...         loss.backward()
            ...         return loss

            >>> linear_graph = LinearTrainGraph()
            >>> x = flow.randn(10, 3)
            >>> y = flow.randn(10)
            >>> for t in range(3):
            ...     loss = linear_graph(x, y)

        参数：
            - **optim** (oneflow.optim.Optimizer): 优化器
            - **lr_sch**: 学习率调整器，请查阅 oneflow.optim.lr_scheduler
        """
)

reset_docstr(
    oneflow.nn.Graph.set_grad_scaler,
    r"""为梯度和损失放缩设置 GradScaler 。
        """
)

reset_docstr(
    oneflow.nn.Graph.__call__,
    r"""调用 nn.Graph 子类实例来运行你的自定义 Graph 。

        实例化后调用你的自定义 Graph ：

        .. code-block:: python

            g = CustomGraph()
            out_tensors = g(input_tensors)

        ``__call__`` 函数的输入必须与 ``build()`` 函数的输入相匹配。
        ``__call__`` 函数将会返回与 ``build()`` 函数输出相匹配的输出。

        请注意，第一次调用会以后的调用花费更长的时间，因为 nn.Graph 将在第一次调用时会
        进行计算图的生成和优化。

        请不要覆盖此函数。
        """
)



reset_docstr(
    oneflow.nn.Graph.debug,
    r"""在 Graph 中打开或关闭 debug 模式。

        如果处于 debug 模式中，将打印计算图构建信息或警告日志。 否则，只会打印错误。

        在 nn.Graph 中的 nn.Module 也有 debug() 函数使得 debug 模式得以运行。

        使用 ``v_level`` 函数来选择详细调试信息级别，默认级别为 0，最大级别为 3。
        ``v_level`` 0 将打印警告和图形构建阶段。 ``v_level`` 1 将另外打印每个 
        nn.Module 的图形构建信息。 ``v_level`` 2 将另外打印每个操作的图形构建信息。
        ``v_level`` 3 将另外打印每个操作的更详细信息。
        
        使用 ``ranks`` 函数来选择要打印调试信息的等级。

        .. code-block:: python

            g = CustomGraph()
            g.debug()  # 打开 debug 模式。
            out_tensors = g(input_tensors)  # 将在第一次调用时打印调试日志。

        参数：
            - **v_level** (int): 选择详细调试信息级别，默认 v_level 为 0，最大 v_level 为 3。
            - **ranks** (int or list(int)): 选择排名以打印调试信息， 默认等级为 ``0`` 。你可以选择任何有效的等级。Ranks 等于 ``-1`` 则表示在所有等级上调试。
            - **mode** (bool): 是否设置调试模式 (``True``) 或者 (``False``)。 默认值： ``True``。
        """
)

reset_docstr(
    oneflow.nn.Graph.__repr__,
    r"""用于打印 Graph 的结构。

        Graph 实例化后打印 Graph 的结构。

        在第一次调用 Graph 后，输入和输出都会被添加进 Graph 结构中。

        .. code-block:: python

            g = CustomGraph()
            print(g)

            out_tensors = g(input_tensors)
            print(g) # 添加了输入和输出信息。

        """
)