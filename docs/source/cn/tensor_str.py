import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.set_printoptions,
    r"""
    设置打印的选项。无耻地取自NumPy的项目。

    参数：
        - **precision** - 浮点输出的精度位数 (默认=4)。
        - **threshold** - 触发总结的数组元素的总数 而不是完全 "repr"（默认=1000）。
        - **edgeitems** - 在每个维度的开始和结束时，汇总的数组项的数量（默认=3）。
        - **linewidth** - 为了插入换行符，每行的字符数（默认 = terminal_columns）。
        - **profile** - 理智的默认值为漂亮的打印。可以用上述任何一个选项来覆盖。(`default`, `short`, `full` 中的任何一个)
        - **sci_mode** - 启用（True）或禁用（False）科学计数法。如果无（默认）被指定，其值由 `oneflow._tensor_str._Formatter` 定义。这个值是由框架自动选择的。

    .. note::
        线宽等于终端列，手动设置将使默认的自动设置失效。
    
    """
)