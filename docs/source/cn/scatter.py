import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.scatter,
    r"""scatter(input, dim, index, src) -> Tensor

    将 `src` 中由 `index` 指定的元素沿维度 `dim` 写入 `input` 中。

    以 3-D tensor 为例，输出被指定为：
    
    .. code-block:: python

        input[index[i][j][k]][j][k] = src[i][j][k]  # 当 dim == 0 时
        input[i][index[i][j][k]][k] = src[i][j][k]  # 当 dim == 1 时
        input[i][j][index[i][j][k]] = src[i][j][k]  # 当 dim == 2 时

    `input` 、 `index` 、 `src` （若为 Tensor ）的维度数必须相同。并且在每个维度 d 上, 
    index.shape(d) <= src.shape(d) 。当维度 d != dim 时， index.shape(d) <= self.shape(d) 。
    注意 `index` 和 `src` 不广播。

    参数：
        - **input** (Tensor): 输入张量
        - **dim** (int): 要索引的维度
        - **index** (Tensor): 要写入的元素的索引张量
        - **src** (Tensor or float): 写入的元素的源张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.ones((3,5))*2
        >>> index = flow.tensor([[0,1,2],[0,1,4]], dtype=flow.int32)
        >>> src = flow.tensor([[0.,10.,20.,30.,40.],[50.,60.,70.,80.,90.]], dtype=flow.float32)
        >>> out = flow.scatter(input, 1, index, src)
        >>> out
        tensor([[ 0., 10., 20.,  2.,  2.],
                [50., 60.,  2.,  2., 70.],
                [ 2.,  2.,  2.,  2.,  2.]], dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.scatter_add,
    r"""scatter_add(input, dim, index, src) -> Tensor
    
    将 `src` 中由 `index` 指定的元素沿维度 `dim` 与 `input` 做加法。

    以 3-D tensor 为例，输出被指定为：
    
    .. code-block:: python

        input[index[i][j][k]][j][k] += src[i][j][k]  # 当 dim == 0 时
        input[i][index[i][j][k]][k] += src[i][j][k]  # 当 dim == 1 时
        input[i][j][index[i][j][k]] += src[i][j][k]  # 当 dim == 2 时

    参数：
        - **input** (Tensor): 输入张量
        - **dim** (int): 要索引的维度
        - **index** (Tensor): 要做加法的元素的索引张量
        - **src** (Tensor or float): 加数的源张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.ones((3,5))*2
        >>> index = flow.tensor([[0,1,2],[0,1,4]], dtype=flow.int32)
        >>> src = flow.tensor([[0,10,20,30,40],[50,60,70,80,90]], dtype=flow.float32)
        >>> out = flow.scatter_add(input, 1, index, src)
        >>> out
        tensor([[ 2., 12., 22.,  2.,  2.],
                [52., 62.,  2.,  2., 72.],
                [ 2.,  2.,  2.,  2.,  2.]], dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.scatter_nd,
    r"""oneflow.scatter_nd(index, update, shape) -> Tensor

    依据 `shape` 创建一个新的元素皆为 0 的 tensor ，并根据 `index` 在新的 tensor 中插入 `update` 的元素。

    参数：
        - **index** (Tensor): 在新 tensor 插入 `update` 的元素时的索引张量。数据类型应为 `oneflow.int` 。
        - **update** (Tensor): 源张量
        - **shape** (Sequence[int]): 常数张量形状，常数张量元素均为 0 。

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> index = flow.tensor([[1], [6], [4]], dtype=flow.int)
        >>> update = flow.tensor([10.2, 5.1, 12.7], dtype=flow.float32)
        >>> out = flow.scatter_nd(index, update, [8])
        >>> out
        tensor([ 0.0000, 10.2000,  0.0000,  0.0000, 12.7000,  0.0000,  5.1000,  0.0000],
               dtype=oneflow.float32)

    """
)
