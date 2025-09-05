import debugpy
import os
import torch.distributed as dist
if False:
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])

        # 只在 rank 0 进程上启动调试服务器
        if local_rank == 0:
            print(f"Rank {local_rank} is starting debugpy on port 5678, waiting for client...")
            # 监听 5678 端口。你可以换成其他端口。
            # host='0.0.0.0' 允许从其他机器（如Docker容器外）连接
            debugpy.listen(('0.0.0.0', 5678)) 
            
            # 暂停程序，直到 VS Code 调试器连接上来
            debugpy.wait_for_client()
            print(f"Rank {local_rank} debugger attached.")
    dist.init_process_group(backend='nccl') # 你的初始化代码
    dist.barrier() 
from llava.train.train import train

if __name__ == "__main__":
    train()
