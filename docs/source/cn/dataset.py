import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.decode_onerec,
    r"""
    (Tensor input, String key, DataType dtype, Shape shape, Bool is_dynamic=False, Shape reshape=None, Shape batch_padding=None)
    decode_onerec(input:Tensor, key:str, dtype, shape, is_dynamic=False, reshape=None, batch_padding=None) -> Tensor

    从应该由 oneflow.nn.OneRecReader 之前生成的输入中解码张量。

    参数:
        - **input** (Tensor) - 之前由 oneflow.nn.OneRecReader 生成的张量。
        - **key** (str) - 要解码的张量的字段名称。
        - **shape** (bool) - 要解码的张量的形状。
        - **is_dynamic** (bool) - 张量形状是否是动态的。
        - **reshape** (tuple) - 设置该参数可以 reshape 张量
        - **batch_padding** (tuple) - 如果需要批量填充，请设置它。

    示例:

    .. code-block:: python

        import oneflow as flow
        files = ['file01.onerec', 'file02.onerec']
        # read onerec dataset form files
        reader = flow.nn.OneRecReader(files, 10, True, "batch")
        readdata = reader()

        # decode
        labels = flow.decode_onerec(readdata, key="labels", dtype=flow.int32, shape=(1,))
        dense_fields = flow.decode_onerec(readdata, key="dense_fields", dtype=flow.float, shape=(13,))

    """
)