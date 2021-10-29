# Oneflow API 中文版本翻译教程

克隆仓库：

```shell
git clone https://github.com/Oneflow-Inc/oneflow-api-cn.git
cd oneflow-api-cn
```

## 首次翻译的准备

如果是首次进行翻译工作，需要运行以下命令（只需运行一次，之后不需要运行）：

1. 安装可以重置 `docstr` 的 Python 包：

```shell
python3 -m pip install -r dev-requirements.txt
python3 setup.py install
```

2. 切换到 `docs` 目录，安装相关依赖：

```shell
cd docs && python3 -m pip install -r requirements.txt
```

3. [安装](https://start.oneflow.org) OneFlow

```shell
python3 -m pip install -f https://staging.oneflow.info/branch/master/cu102 oneflow
```

4. 在 `master` 分支上创建新的分支，开始翻译工作：

```shell
git checkout -m <本地分支名>
```

## 查找文本

克隆 oneflow 源码：

```shell
git clone https://github.com/Oneflow-Inc/oneflow.git
```

如果你在使用 vscode ，使用 `文件` 中的 `打开文件夹` 打开 oneflow 文件夹。然后用 `编辑` 中的 `在文件中查找` 查找关键词即可找到英文文本。

注意：此步中最好记录下英文文本在源代码中的位置，方便后续提交与审核。

## 开始翻译

在 [docs/source/cn](./docs/source/cn) 下对应的文件中，通过调用 `docreset.reset_docstr`，把原有的 `__doc__` 替换为中文翻译。

### r 字符串

翻译的文本应使用 r 字符串 (r"""...""") ， r 字符串与普通字符串的区别在于在 r 字符串中的 `\` 不是转义符，如 `\n` 在一般字符串中代表换行符，但是如果有前缀 r 那么就不转义这个 `\` ，则 r"""\n""" 表示一个普通字符串 \n ，而不是换行。 r 字符串中的换行为 `\\` 。

注意如果在源码中的文本为普通字符串而不是 r 字符串，在改为 r 字符串时要将原字符串中连续两个反斜杠 `\\` 改为 `\` 。

### 测试代码

新版本的 oneflow 已不依赖 numpy 库，故在翻译过程中应去掉示例中的 numpy 算子，并改为 oneflow 中的同功能算子。

例如：

```python
>>> import numpy as np
>>> import oneflow as flow
>>> x = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
>>> y = flow.tensor(np.array([1, 2, 3]), dtype=flow.float32)
>>> y.numpy()
array([1., 2., 3.], dtype=float32)
```

应改为

```python
>>> import oneflow as flow
>>> x = flow.randn(2 , 3, dtype=flow.float32)
>>> y = flow.tensor([1, 2, 3], dtype=flow.float32)
>>> y
tensor([1., 2., 3.], dtype=oneflow.float32)
```

### 翻译格式

建议在 api 文本正文中出现的参数为 :attr:\`\` 格式，如一个参数名为 input ，则在中文文本正文中出现的 input 可以表示为 :attr:\`input\` 。

在 oneflow 源码中的英文文本部分关键词会被自动转义，例如：

关键词 `Args:` 将期内容冒号前转化为黑体，并分行，将冒号 `:` 转化为 `-` 。而中文版无法自行转化此类效果。所以应手动操作：

英文版：

```python
r"""
    Args:
        x (oneflow.Tensor): Input Tensor
        dtype (flow.dtype): Data type of the output tensor
"""
```

中文版：

```python
r"""
    参数：
        - **x** (oneflow.Tensor) - 输入张量
        - **dtype** (flow.dtype) - 输出张量的数据类型
"""
```

英文版文档中的 `Return Type` 中部分使用了 oneflow.Tensor ，应该为 oneflow.tensor ，具体原因请参考 https://docs.oneflow.org/master/basics/02_tensor.html#tensor-tensor

### 关于参数

Oneflow 的参数原则上应与 Pytorch 对齐，所以翻译过程中应该与 https://pytorch.org/docs/stable/index.html 中的算子对照，并将未对其的算子汇报到 https://github.com/Oneflow-Inc/oneflow/issues/6003 。
Oneflow 源码中英文文版有部分参数出错的问题，建议翻译前先使用
```shell
python3
```
进行试错，如发现错误请在中文文档中改正并提交至此 issue ：


另外，如发现其他问题，也可以提交至此 issue。

### 以下为一个标准的翻译示例：

英文版:

```python
add_docstr(
    oneflow.asin,
    r"""
    Returns a new tensor with the arcsine of the elements of :attr:`input`.

    .. math::
        \\text{out}_{i} = \\sin^{-1}(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.tensor(np.array([-0.5,  0.8, 1.0,  -0.8]), dtype=flow.float32)
        >>> output = flow.asin(input)
        >>> output.shape
        oneflow.Size([4])
        >>> output
        tensor([-0.5236,  0.9273,  1.5708, -0.9273], dtype=oneflow.float32)
        >>> input1 = flow.tensor(np.array([[0.8, 1.0], [-0.6, -1.0]]), dtype=flow.float32)
        >>> output1 = input1.asin()
        >>> output1.shape
        oneflow.Size([2, 2])
        >>> output1
        tensor([[ 0.9273,  1.5708],
                [-0.6435, -1.5708]], dtype=oneflow.float32)
    """,
)
```

中文版:

```python
reset_docstr(
    oneflow.asin,
    r"""arcsin(x) -> Tensor

    返回一个新的 tensor 包含 :attr:`x` 中每个元素的反正弦。

    公式为：

    .. math::
        \text{out}_{i} = \sin^{-1}(\text{input}_{i})

    参数：
        **x** (Tensor) - 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([-0.5,  0.8, 1.0,  -0.8], dtype=flow.float32)
        >>> output = flow.asin(input)
        >>> output.shape
        oneflow.Size([4])
        >>> output
        tensor([-0.5236,  0.9273,  1.5708, -0.9273], dtype=oneflow.float32)
        >>> input1 = flow.tensor([[0.8, 1.0], [-0.6, -1.0]], dtype=flow.float32)
        >>> output1 = input1.asin()
        >>> output1.shape
        oneflow.Size([2, 2])
        >>> output1
        tensor([[ 0.9273,  1.5708],
                [-0.6435, -1.5708]], dtype=oneflow.float32)
    """,
)
```

如果翻译过程中在 [docs/source/cn](./docs/source/cn) 文件夹中添加了新的 .py 文件，则需要在 \_init_.py 中添加新增的文件才能查看翻译效果以及测试代码。

## 查看效果

可以编译文档并下载整个 [docs/build-cn/html](./docs/build-cn/html) 文件夹到本地，查看效果：

```shell
cd docs && make html_cn
```

测试文档内 `.. code-block:: python` 下的代码：

```shell
cd docs && make test_cn
```

如果想对具体代码问题进行测试可以运行 Python ，然后逐行运行代码，则会显示具体输出：

```shell
python3
```

### 常见问题

1. WARNING: Inline strong start-string without end-string.
解决方法：注意是否前后 * 、 \` 数量一致，并在最后一个 * 或 \` 后面有空格。如果没有空格会导致此错误。

## 提交翻译

如果在 `docs` 下运行  
```shell
make test_cn
```
和
```shell
make html_cn
```
后未出现 warning ，则可以提交翻译进度。

查看哪些文件被修改过：

```shell
git status
```

对所有要提交的文件执行以下代码，将文件添加到暂存区:

```shell
git add [file1] [file2] ...
```

将暂存区内容添加到本地仓库中， \[message] 为要备注的信息:

```shell
git commit -m [message]
```

从将本地的分支版本上传到远程：

```shell
git push origin <本地分支名>
```

如果是第一次 push ，则需要在 github 新建 pr 。新 pr 名应该为英文。

每合并一个 pr 后，应该同步 `master` ，
```shell
git checkout master
git pull origin master
```
再从 `master` 新建 branch 后在新建的 branch 中进行翻译。
```shell
git checkout -m <本地新分支名>
```

提交翻译时应附上中文版算子的网页截图以及英文版文本在 oneflow 仓库(https://github.com/Oneflow-Inc/oneflow) 中位置的链接以方便审核。
