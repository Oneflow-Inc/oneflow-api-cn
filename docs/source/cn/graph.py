import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.nn.Graph,
    r"""在 Graph 模块下训练和测试一个神经网络的基类。

    要在 OneFlow 中使用 Graph 模块进行模型训练或测试，你应该：

    1. 将你的自定义 Graph 定义为 ``nn.Graph`` 的子类。
    2. 在子类 ``__init__()`` 中添加 ``super().__init__()`` 。
    3. 将模块作为常规属性添加到 Graph 中。
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

        # 第一次调用 Graph 将运行 graphs build() 函数来跟踪计算图。 
        # 计算图将首次执行并被优化。
        >>> linear_graph(x).shape
        oneflow.Size([4, 8])

        # 然后调用 Graph 将直接运行计算图。
        >>> linear_graph(x).shape
        oneflow.Size([4, 8])

    请注意：
         Graph 目前不可以进行嵌套。
    """
)

reset_docstr(
    oneflow.nn.Graph.__init__,
    r"""
        初始化 Graph 的内部状态。 它必须在子类的 ``__init__`` 函数中调用。

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
    r"""必须重写 ``build()`` 函数来定义神经网络计算逻辑。

        nn.Graph 中的 ``build()`` 函数与 nn.Module 中的 ``forward()`` 函数非常相似。它是用来描述神经网络的计算逻辑。

        当第一次调用 Graph 对象时，会隐式调用 ``build()`` 函数来构建计算图。

        确保在第一次调用 Graph 之前先调用模块中的 ``train()`` 或 ``eval()`` 函数，以使 Graph 执行正确的训练或测试模式。

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

        请注意:
            ``build()`` 函数的输入和输出目前支持列表/元组/字典，但其中的对象必须是以下类型之一：

            * ``Tensor``
            * ``None``

        """
)

reset_docstr(
    oneflow.nn.Graph.add_optimizer,
    r"""向 Graph 中添加一个优化器，一个学习率调整器。

        要使用 nn.Graph 进行训练，你应该再做 2 件事：

        1. 使用 ``add_optimizer()`` 函数添加至少一个优化器（ 学习率调整器是可选的 ）。
        2. 在 ``build()`` 函数中调用 loss tensor 的 ``backward()`` 函数。

        请注意，计算图将自动执行这些方法： 

        * 如果优化器设置为梯度裁剪，则调用 ``clip_grad()`` 函数。
        * 优化器的 ``step()`` 函数。
        * 优化器的 ``zero_grad()`` 函数。
        * 学习率调整器的 ``step()`` 函数。

        另请注意，在 ``nn.Graph.build()``  中暂时只允许 ``backward()`` 调用标量张量。 
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
            ...         # 调用 loss tensor 中的 backward() 函数, loss tensor 必须是一个标量张量。
            ...         loss.backward()
            ...         return loss

            >>> linear_graph = LinearTrainGraph()
            >>> x = flow.randn(10, 3)
            >>> y = flow.randn(10)
            >>> for t in range(3):
            ...     loss = linear_graph(x, y)

        参数：
            - **optim** (oneflow.optim.Optimizer) - 优化器
            - **lr_sch** - 学习率调整器，参见 oneflow.optim.lr_scheduler
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

        实例化后调用 Graph ：

        .. code-block:: python

            g = CustomGraph()
            out_tensors = g(input_tensors)

        ``__call__`` 函数的输入必须与 ``build()`` 函数的输入相匹配。
        ``__call__`` 函数将会返回与 ``build()`` 函数输出相匹配的输出。

        请注意:
            第一次调用会比之后的调用花费时间更长，因为 nn.Graph 在第一次调用时会进行计算图的生成和优化。

            请不要覆盖此函数。

        """
)



reset_docstr(
    oneflow.nn.Graph.debug,
    r"""在 Graph 中打开或关闭 debug 模式。

        如果处于 debug 模式中，将打印计算图的构建信息或警告日志。 否则，只会打印错误。

        在 nn.Graph 中的每个 nn.Module 也有 debug() 函数使得 debug 模式得以运行。

        使用 ``v_level`` 函数来选择详细的 debug 信息级别，默认级别为 0，最大级别为 3。``v_level`` 0 将打印警告和 Graph 构建过程。 ``v_level`` 1 将另外打印每个 nn.Module 的 Graph 构建信息。 ``v_level`` 2 将另外打印每个操作的 Graph 构建信息。``v_level`` 3 将另外打印每个操作的更详细信息。
        
        使用 ``ranks`` 函数来选择要打印 debug 信息的 rank。

        .. code-block:: python

            g = CustomGraph()
            g.debug()  # 打开 debug 模式。
            out_tensors = g(input_tensors)  # 将在第一次调用时打印调试日志。

        参数：
            - **v_level** (int)- 选择详细的 debug 信息级别，默认 v_level 为 0，最大 v_level 为 3。
            - **ranks** (int or list(int))- 选择 ranks 以打印 debug 信息， 默认 rank 为 ``0`` 。你可以选择任何有效的 rank 。Ranks 等于 ``-1`` 则表示在所有 ranks 上调试。
            - **mode** (bool)- 设置调试模式运行 (``True``) 或者停止 (``False``)。 默认值： ``True``。
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

reset_docstr(
    oneflow.nn.Graph.name,
    r"""为该 Graph 自动生成的名称。
        """)

reset_docstr(
    oneflow.nn.Graph.training,
    r"""如果 Graph 有优化器则处于训练模式。
        """
)

reset_docstr(
    oneflow.nn.Graph.state_dict,
    r"""返回包含 Graph 所有属性的字典。

        包含 Graph 中模块/优化器/学习率调整器的属性。

        模块属性字典的键值与它们在 Graph 中的名称相对应。
        模块属性字典的值与它们的 nn.Module 的属性字典相对应。

        其他键值或张量是优化器/学习率调整器等的属性。

        返回:
            一个包含 Graph 所有属性的字典。 
        
        返回类型：
            dict

        """
)

reset_docstr(
    oneflow.nn.Graph.load_state_dict,
    r"""用 :attr:`state_dict` 复制这个模块的属性和其他 Graph 的属性到这个 Graph 中。如果 :attr:`strict` 的值是 ``True``， 那么 :attr:`state_dict` 的键值必须和返回的键值通过该模块的 :meth:`nn.Graph.state_dict` 函数精确匹配。

        参数:
            - **state_dict** (dict)- 一个包含模块所有属性和其他 Graph 属性的字典。
            - **strict** (bool, optional)- 是否严格执行使 :attr:`state_dict` 中的键值与在 Graph 中 :meth:`nn.Graph.state_dict` 函数返回的键值相匹配。默认值是 ``True``.

        请注意:
            nn.Graph 的属性字典只能在第一次调用 Graph 前被加载。
        """
)

reset_docstr(
    oneflow.nn.graph.graph_config.GraphConfig,
    r"""
    用于 nn.Graph 的配置。
    """
)

reset_docstr(
    oneflow.nn.graph.graph_config.GraphConfig.enable_amp,
    r"""
    enable_amp(mode)

    如果设置为 True ，graph 会使用混合的精度模式，即在模型训练中同时使用 float16 和 float32。

        示例：

        .. code-block:: python

            import oneflow as flow

            class Graph(flow.nn.Graph):
                def __init__(self):
                    super().__init__()
                    self.linear = flow.nn.Linear(3, 8, False)
                    self.config.enable_amp(True) # Use mixed precision mode.
                def build(self, x):
                    return self.linear(x)

            graph = Graph()

        参数：
            - **mode** (bool, 可选): 默认值为 True 。
        """
)

reset_docstr(
    oneflow.nn.graph.graph_config.GraphConfig.allow_fuse_model_update_ops,
    r"""
    allow_fuse_model_update_ops(mode)

        如果设置为 True ，将尝试融合 cast + scale + l1_l2_regularize_gradient + model_update 为一次操作以提升性能。

        示例：

        .. code-block:: python

            import oneflow as flow

            class Graph(flow.nn.Graph):
                def __init__(self):
                    super().__init__()
                    self.linear = flow.nn.Linear(3, 8, False)
                    self.config.allow_fuse_model_update_ops(True)
                def build(self, x):
                    return self.linear(x)

            graph = Graph()

        参数：
            - **mode** (bool, 可选): 默认值为 True 。
        """
)

reset_docstr(
    oneflow.nn.graph.graph_config.GraphConfig.allow_fuse_add_to_output,
    r"""
    allow_fuse_add_to_output(mode)

        如果设置为 True，将尝试融合一个二进制 element-wise add 运算符进入前置算子以提升性能。

        示例：

        .. code-block:: python

            import oneflow as flow

            class Graph(flow.nn.Graph):
                def __init__(self):
                    super().__init__()
                    self.bn1 = flow.nn.BatchNorm1d(100)
                    self.config.allow_fuse_add_to_output(True)
                def build(self, x):
                    bn = self.bn1(x) 
                    out = bn + x
                    return out

            graph = Graph()

        参数：
            - **mode** (bool, 可选): 默认值为 True 。
        """
)

reset_docstr(
    oneflow.nn.graph.graph_config.GraphConfig.allow_fuse_cast_scale,
    r"""
    allow_fuse_cast_scale(mode)

        如果设置为 True，将尝试融合 cast 和 scalar_mul_by_tensor 以提升性能。
    
        示例：

        .. code-block:: python

            import oneflow as flow

            def model(x):
                return flow.mul(1,flow.cast(x,flow.int8))

            class Graph(flow.nn.Graph):
                def __init__(self):
                    super().__init__()
                    self.m=model
                    self.config.allow_fuse_cast_scale(True)
                def build(self, x):
                    return self.m(x)

            graph = Graph()

        参数：
            - **mode** (bool, 可选): 默认值为 True 。
        """
)
reset_docstr(
    oneflow.nn.graph.graph_config.GraphConfig.set_gradient_accumulation_steps,
    r"""
    set_gradient_accumulation_steps(value)

    设置累加梯度的步数。

        示例：

        .. code-block:: python

            import oneflow as flow

            class Graph(flow.nn.Graph):
                def __init__(self):
                    super().__init__()
                    self.linear = flow.nn.Linear(3, 8, False)
                    # Let graph do gradient accumulation, such as pipelining parallelism depends on gradient accumulation.
                    self.config.set_gradient_accumulation_steps(4)
                def build(self, x):
                    return self.linear(x)

            graph = Graph()

        参数：
            - **value** (int): 步数的数量。
        """
)

reset_docstr(
    oneflow.nn.graph.graph_config.GraphConfig.set_zero_redundancy_optimizer_mode,
    r"""
    set_zero_redundancy_optimizer_mode(mode)

        设置模式以移除冗余优化步骤。
        此优化将根据 ZeRO https://arxiv.org/abs/1910.02054 的描述来减少优化步骤的内存消耗。

        示例：

        .. code-block:: python

            import oneflow as flow

            class Graph(flow.nn.Graph):
                def __init__(self):
                    super().__init__()
                    self.linear = flow.nn.Linear(3, 8, False)
                    self.config.set_zero_redundancy_optimizer_mode("distributed_split")
                def build(self, x):
                    return self.linear(x)

            graph = Graph()

        参数：
            - **mode** (str): "distributed_split" 或 "non_distributed" 。 "distributed_split" 模式将把每个优化步骤分散到每个设备，而 "non_distributed" 将把每个优化步骤放到一个设备上。
        """
)

reset_docstr(
    oneflow.nn.graph.graph_config.GraphConfig.set_zero_redundancy_optimizer_min_size_after_split,
    r"""
    set_zero_redundancy_optimizer_min_size_after_split(value)
    
    设置切分后优化器步骤/梯度/参数的最小值。

        示例：

        .. code-block:: python

            import oneflow as flow

            class Graph(flow.nn.Graph):
                def __init__(self):
                    super().__init__()
                    self.linear = flow.nn.Linear(3, 8, False)
                    self.config.set_zero_redundancy_optimizer_mode("distributed_split")
                    self.config.set_zero_redundancy_optimizer_min_size_after_split(1)
                def build(self, x):
                    return self.linear(x)

            graph = Graph()

        Args:
            - **value** (int): 最小值。
        """
)

reset_docstr(
    oneflow.nn.graph.graph_config.GraphConfig.enable_xla_jit,
    r"""
    enable_xla_jit(value)

        是否在 xrt 中使用 xla_jit。

        启用此选项时，oneflow 将检查所有算子是否被 xla_jit 支持，将支持的算子聚为子图，并用 xla_jit 运算子图。
           XLA: https://www.tensorflow.org/xla

        如果需要使用 XLA 来优化模型运行速度，则需要编译 XLA 版本的 oneflow 。
        
        使用 XLA 构建 oneflow 的教程：
        
        https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/xrt/README.md#build-with-xla

        示例：

        .. code-block:: python

            import oneflow as flow

            class Graph(flow.nn.Graph):
                def __init__(self):
                    super().__init__()
                    self.linear = flow.nn.Linear(3, 8, False)
                    self.config.enable_xla_jit(True) # 在 xrt 中使用 xla_jit 。
                def build(self, x):
                    return self.linear(x)

            graph = Graph()

        参数：
            - **value** (bool, 可选): 默认值为 True.
        """
)

reset_docstr(
    oneflow.nn.graph.graph_config.GraphConfig.enable_tensorrt,
    r"""
    enable_tensorrt(value)

        是否在 xrt 中使用 tensorrt。

        启用此选项时，oneflow 将检查所有算子是否被 tensorrt 支持，将支持的算子聚为子图，并用 xla_jit 运算子图。

        TensorRT: https://developer.nvidia.com/tensorrt

        如果需要使用 XLA 来优化模型运行速度，则需要编译 TensorRT 版本的 oneflow 。

        使用 TensorRT 构建 oneflow 的教程：
        
        https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/xrt/README.md#build-with-tensorrt

        示例：

        .. code-block:: python

            import oneflow as flow

            class Graph(flow.nn.Graph):
                def __init__(self):
                    super().__init__()
                    self.linear = flow.nn.Linear(3, 8, False)
                    self.config.enable_tensorrt(True) # Use tensorrt in xrt.
                def build(self, x):
                    return self.linear(x)

            graph = Graph()

        参数：
            - **value** (bool, 可选): 默认值为 True.
        """
)

reset_docstr(
    oneflow.nn.graph.graph_config.GraphConfig.enable_openvino,
    r"""
    enable_openvino(value)

        是否在 xrt 中使用 openvino。

        启用此选项时，oneflow 将检查所有算子是否被 openvino 支持，将支持的算子聚为子图，并用 xla_jit 运算子图。

        请注意，openvino 仅支持引用模式。
        OpenVINO: https://developer.nvidia.com/tensorrt

        如果需要使用 XLA 来优化模型运行速度，则需要编译 TensorRT 版本的 oneflow 。

        同时也需要编译 XLA 或者 TensorRT 版本的 oneflow ,教程见下：
        
        https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/xrt/README.md#build-with-tensorrt

        示例：

        .. code-block:: python

            import oneflow as flow

            class Graph(flow.nn.Graph):
                def __init__(self):
                    super().__init__()
                    self.linear = flow.nn.Linear(3, 8, False)
                    self.config.enable_tensorrt(True) # Use tensorrt in xrt.
                def build(self, x):
                    return self.linear(x)

            graph = Graph()

        参数：
            - **value** (bool, 可选): 默认值为 True.
        """
)

reset_docstr(
    oneflow.nn.graph.graph_config.GraphConfig.enable_cudnn_conv_heuristic_search_algo,
    r"""
    enable_cudnn_conv_heuristic_search_algo(value)
    
        是否启用 cudnn conv 操作来使用启发式搜索算法。

        示例：

        .. code-block:: python

            import oneflow as flow

            class Graph(flow.nn.Graph):
                def __init__(self):
                    super().__init__()
                    self.m = flow.nn.Conv2d(16, 32, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
                    # Do not enable the cudnn conv operation to use the heuristic search algorithm.
                    self.config.enable_cudnn_conv_heuristic_search_algo(False)
                def build(self, x):
                    return self.m(x)

            graph = Graph()
    
        参数：
            - **value** (bool, 可选): 默认值为 True.
        """
)

reset_docstr(
    oneflow.nn.graph.block_config.BlockConfig,
    r"""nn.Graph ModuleBlock 的配置。

    当一个 nn.Module 被加至 nn.Graph 时，他会被包络在一个 ModuleBlock 中。你可以在一个 nn.Module 使用 `ModuleBlock.config` 设置或获取优化参数。
    """
)

reset_docstr(
    oneflow.nn.graph.block_config.BlockConfig.stage_id,
    r"""设置/获取 nn.Module/ModuleBlock 在并行管线中的 stage id 。
        
        在调用 stage_id(value: int = None) 时，将设置不同 module 的 id 以使得 graph 在管线中准备正确数量的缓冲。

        示例：

        .. code-block:: python

            # m_stage0 和 m_stage1 是网络的两个管线阶段。
            # 我们可以通过设置 config.stage_id 属性来设置 Stage ID 。
            # Stage ID 从 0 开始以整数计数。
            self.module_pipeline.m_stage0.config.stage_id = 0
            self.module_pipeline.m_stage1.config.stage_id = 1

        """

)

reset_docstr(
    oneflow.nn.graph.block_config.BlockConfig.activation_checkpointing,
    r"""设置/获取是否在此 nn.Module 中执行 activation checkpointing 。

        示例：

        .. code-block:: python

            import oneflow as flow

            class Graph(flow.nn.Graph):
                def __init__(self):
                    super().__init__()
                    self.linear1 = flow.nn.Linear(3, 5, False)
                    self.linear2 = flow.nn.Linear(5, 8, False)
                    self.linear1.config.activation_checkpointing = True
                    self.linear2.config.activation_checkpointing = True

                def build(self, x):
                    y_pred = self.linear1(x)
                    y_pred = self.linear2(y_pred)
                    return y_pred

            graph = Graph()

        """

)