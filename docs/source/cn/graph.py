import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.nn.Graph,
    r"""在 Graph 模块下训练和评估一个神经网络的基类。

    要在 OneFlow 中使用 Graph 模块进行模型训练或评估，你应该：

    1. 将你的自定义 Graph 定义为 ``nn.Graph``的子类。
    2. 在子类 ``__init__()``中添加``super().__init__()``。
    3. 将模块作为常规属性添加到你的Graph中。
    4. 在 ``build()`` 中定义计算逻辑。
    5. 将 Graph 实例化然后调用它。

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