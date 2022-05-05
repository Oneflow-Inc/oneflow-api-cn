oneflow.distributed
=========================================================

.. currentmodule:: oneflow.distributed

运行以下命令来了解更多用法。
::

    python3 -m oneflow.distributed.launch -h

.. code-block::

    usage: launch.py [-h] [--nnodes NNODES] [--node_rank NODE_RANK]
                 [--nproc_per_node NPROC_PER_NODE] [--master_addr MASTER_ADDR]
                 [--master_port MASTER_PORT] [-m] [--no_python]
                 [--redirect_stdout_and_stderr] [--logdir LOGDIR]
                 training_script ...

    OneFlow 分布式训练启动协助功能，将启动多个分布式进程。

    位置参数：
    training_script       和进程同时启动的单 GPU 训练程序/脚本的完整路径，跟随着训练脚本的所有参数。
    training_script_args

    optional arguments:
    -h, --help            展现此帮助页面并退出。
    --nnodes NNODES       用于分布式训练的节点数量。
    --node_rank NODE_RANK
                            multi-mode 分布式训练中节点的 rank 。
    --nproc_per_node NPROC_PER_NODE
                            每个节点上启动的进程数量，对于 GPU 训练，推荐将此参数设置为 GPU 数量，
                            这样每个进程都能绑定一个单独的 GPU 。
    --master_addr MASTER_ADDR
                            主节点 (rank 0) 的地址，应为主节点的 IP 地址和主机地址其中之一。
                            对于单节点多进程训练， --master_addr 可直接设为 127.0.0.1。
    --master_port MASTER_PORT
                            主节点 (rank 0) 的空余 port ，将被用于分布式训练中各节点的联系。
    -m, --module            
                            使每个进程将脚本解释为 python 模组，和使用 'python -m' 运行脚本有相同效果。
    --no_python             
                            不对训练脚本前置 "python" ，而是直接运行。在脚本不是 Python 脚本时有用。
    --redirect_stdout_and_stderr
                            对文件 'stdout' 和 'stderr' 输出 stdout 和 stderr 。只有在 logdir 被设定后可用。
    --logdir LOGDIR         
                            子进程输出 log 文件的相对位置。如果需要，将根据相对位置创建新路径。
                            需注意，连续使用相同的路径运行多次程序会覆盖之前的 log ，所以记得在需要时保存 log 。