.. currentmodule:: oneflow

Tensor Attributes
=============================================================
每个本地 ``oneflow.Tensor`` 都有一个 :class:`oneflow.dtype` 和 :class:`oneflow.device` ，而每个全局 ``oneflow.Tensor`` 都有一个 :class:`oneflow.dtype`, :class:`oneflow.placement` 和 :class:`oneflow.sbp` 。

oneflow.device
--------------------------------------------------------------
.. autoclass:: oneflow.device

oneflow.placement
--------------------------------------------------------------
.. autoclass:: oneflow.placement

oneflow.env.all_device_placement
--------------------------------------------------------------
.. autofunction:: oneflow.env.all_device_placement

oneflow.sbp.sbp
--------------------------------------------------------------
.. autoclass:: oneflow.sbp.sbp
