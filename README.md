克隆仓库：

```shell
git clone https://github.com/Oneflow-Inc/oneflow-api-cn.git
cd oneflow-api-cn
```

## 首次翻译的准备

如果是首次进行翻译工作，需要运行以下命令（只需运行一次，之后不需要运行）：

1. 安装相关依赖：

```shell
python3 -m pip install -r docs/requirements.txt
```

2. 运行 `setup.py`，安装中文版需要的 `docreset` 包

```shell
python3 setup.py install
```

3. [安装](https://start.oneflow.org) OneFlow

```shell
python3 -m pip install -f https://staging.oneflow.info/branch/master/cu102 oneflow
```

## 开始翻译

在 [docs/source/cn](./docs/source/cn) 下对应的文件中，通过调用 `docreset.reset_docstr`，把原有的 `__doc__` 替换为中文翻译。

```python
reset_docstr(
    oneflow.add,
    r"""add(input, other)

    计算 `input` 和 `other` 的和。支持 element-wise、标量和广播形式的加法。
    公式为：

    .. math::
        out = input + other

    示例：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        # element-wise 加法
        >>> x = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> y = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.add(x, y).numpy()
        >>> out.shape
        (2, 3)

        # 标量加法
        >>> x = 5
        >>> y = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.add(x, y).numpy()
        >>> out.shape
        (2, 3)

        # 广播加法
        >>> x = flow.tensor(np.random.randn(1,1), dtype=flow.float32)
        >>> y = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.add(x, y).numpy()
        >>> out.shape
        (2, 3)

    """,
)
```

## 查看效果

可以在本地编译文档，查看效果：

```shell
cd docs && make html
```
