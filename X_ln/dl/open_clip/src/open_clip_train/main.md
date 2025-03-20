Lan: `py` From`dl/open_clip/src/open_clip_train/main.py`


**1. 导入必要的库 (导入包)**

```python
import copy
import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime
from functools import partial

import numpy as np
import torch
from torch import optim

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from open_clip_train.data import get_data
from open_clip_train.distributed import is_master, init_distributed_device, broadcast_object
from open_clip_train.logger import setup_logging
from open_clip_train.params import parse_args
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown
from open_clip_train.train import train_one_epoch, evaluate
from open_clip_train.file_utils import pt_load, check_exists, start_sync_process, remote_sync
```

*   **描述:** 这段代码导入了所有需要的Python库。例如 `torch` 用于深度学习，`numpy` 用于数值计算，`os` 用于文件系统操作等等。`try...except` 块处理了可选库（如 `wandb`, `tensorboard`, `horovod`）的导入，如果这些库没有安装，程序会继续运行，只是相关的功能会失效。`open_clip` 和 `open_clip_train` 导入的是CLIP模型和训练相关的代码。
*   **使用说明:** 这些库是整个程序运行的基础。你需要确保它们都正确安装。
*   **示例:** 无需单独运行，它们会被后续的代码所调用。

**2. 设置随机种子 (设置随机数种子)**

```python
def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
```

*   **描述:** 这个函数用于设置随机种子，以确保实验的可重复性。对于分布式训练，`rank` 参数用于为每个进程设置不同的种子，避免所有进程都使用相同的随机数序列。
*   **使用说明:** 在程序开始时调用一次，例如 `random_seed(args.seed, args.rank)`，`args.seed` 是用户指定的随机种子，`args.rank` 是当前进程的rank。
*   **示例:**

    ```python
    random_seed(42, 0) # 设置种子为42，rank为0
    print(torch.rand(1)) # 每次运行都会得到相同的结果
    ```

**3. 获取最新的检查点 (获取最新的模型checkpoint)**

```python
def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None
```

*   **描述:** `get_latest_checkpoint` 函数用于在指定的路径中查找最新的检查点文件（`.pt` 文件）。它支持本地文件系统和云存储（如 AWS S3）。`natural_key` 函数用于对文件名进行自然排序，确保 `epoch_10.pt` 排在 `epoch_2.pt` 之后。
*   **使用说明:** 在程序启动时调用，如果 `args.resume == 'latest'`，则使用此函数查找最新的检查点，以便从上次训练中断的地方继续训练。
*   **示例:**

    ```python
    checkpoint_path = "/path/to/checkpoints"
    latest_checkpoint = get_latest_checkpoint(checkpoint_path, remote=False)
    if latest_checkpoint:
        print(f"找到最新的检查点: {latest_checkpoint}")
    else:
        print("未找到检查点")
    ```

**4. 主函数 (主函数)**

```python
def main(args):
    args = parse_args(args)

    # ... (省略部分代码)

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
    
        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        opt = getattr(args, 'opt', 'adamw').lower()
        if opt.startswith('timm/'):
            from timm.optim import create_optimizer_v2
            timm_opt = opt.split('timm/')[-1]
            opt_kwargs = {}
            assert (args.beta1 is None) == (args.beta2 is None), \
                'When using timm optimizer, BOTH beta1 and beta2 must be specified (or not specified).'
            if args.beta1 is not None:
                opt_kwargs['betas'] = (args.beta1, args.beta2)
            if args.momentum is not None:
                opt_kwargs['momentum'] = args.momentum
            optimizer = create_optimizer_v2(
                model,
                timm_opt,
                lr=args.lr,
                weight_decay=args.wd,
                eps=args.eps,
                **opt_kwargs,
            )
        else:
            # If some params are not passed, we use the default values based on model name.
            exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
            include = lambda n, p: not exclude(n, p)

            named_parameters = list(model.named_parameters())
            gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
            rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

            if opt == 'adamw':
                optimizer = optim.AdamW(
                    [
                        {"params": gain_or_bias_params, "weight_decay": 0.},
                        {"params": rest_params, "weight_decay": args.wd},
                    ],
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                )
            else:
                assert False, f'Unknown optimizer {opt}'

        if is_master(args):
            if is_master(args):
                defaults = copy.deepcopy(optimizer.defaults)
                defaults['weight_decay'] = args.wd
                defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
                logging.info(
                    f'Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}'
                )

        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = None
        if args.precision == "amp":
            try:
                scaler = torch.amp.GradScaler(device=device)
            except (AttributeError, TypeError) as e:
                scaler = torch.cuda.amp.GradScaler()

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize datasets
    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir)
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None,\
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info('Compiling model...')

        if args.grad_checkpointing and args.distributed:
            logging.info('Disabling DDP dynamo optimizer when grad checkpointing enabled.')
            # As of now (~PyTorch 2.4/2.5), compile + grad checkpointing work, but DDP optimizer must be disabled
            torch._dynamo.config.optimize_ddp = False

        model = torch.compile(original_model)

    if 'train' not in data:
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None:
            from open_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(model)
        # Evaluate.
        evaluate(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer)
        return

    loss = create_loss(args)

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
        completed_epoch = epoch + 1

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            evaluate(model, data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer)

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')
    

def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
```

*   **描述:** `main` 函数是程序的主入口。它负责：

    *   解析命令行参数 (`parse_args`)。
    *   初始化分布式环境 (`init_distributed_device`)。
    *   创建模型和数据预处理 (`create_model_and_transforms`, `get_data`)。
    *   创建优化器和学习率调度器。
    *   加载检查点（如果指定了 `args.resume`）。
    *   设置日志记录和报告（如 TensorBoard 和 WandB）。
    *   进行训练和验证循环 (`train_one_epoch`, `evaluate`)。
    *   保存检查点。
*   **使用说明:**  这是整个程序的控制中心。通过命令行参数控制程序的行为。
*   **示例:**  要运行程序，你需要提供一些命令行参数，例如：

    ```bash
    python your_script.py --train-data /path/to/train_data --val-data /path/to/val_data --lr 1e-4 --batch-size 32 --epochs 10
    ```

    这将使用指定的数据集、学习率、批次大小和训练轮数来运行训练。

**5. 分布式训练 (分布式训练)**

```python
    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
    
        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)
```

*   **描述:** 这段代码处理分布式训练的设置。如果启用了分布式训练 (`args.distributed`) 并且未使用 Horovod (`not args.horovod`)，它会使用 `torch.nn.parallel.DistributedDataParallel` (DDP) 将模型包装起来，以便在多个 GPU 或节点上进行训练。 `SyncBatchNorm` 用于在分布式训练中同步批归一化层。
*   **使用说明:** 运行分布式训练需要使用 `torch.distributed.launch` 或类似的工具启动多个进程。
*   **示例:**

    ```bash
    python -m torch.distributed.launch --nproc_per_node=8 your_script.py --distributed --train-data ...
    ```

    这将启动 8 个进程，每个进程在一个 GPU 上运行。

**6. 优化器和学习率调度器 (优化器和学习率调整器)**

```python
    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        opt = getattr(args, 'opt', 'adamw').lower()
        if opt.startswith('timm/'):
            from timm.optim import create_optimizer_v2
            timm_opt = opt.split('timm/')[-1]
            opt_kwargs = {}
            assert (args.beta1 is None) == (args.beta2 is None), \
                'When using timm optimizer, BOTH beta1 and beta2 must be specified (or not specified).'
            if args.beta1 is not None:
                opt_kwargs['betas'] = (args.beta1, args.beta2)
            if args.momentum is not None:
                opt_kwargs['momentum'] = args.momentum
            optimizer = create_optimizer_v2(
                model,
                timm_opt,
                lr=args.lr,
                weight_decay=args.wd,
                eps=args.eps,
                **opt_kwargs,
            )
        else:
            # If some params are not passed, we use the default values based on model name.
            exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
            include = lambda n, p: not exclude(n, p)

            named_parameters = list(model.named_parameters())
            gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
            rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

            if opt == 'adamw':
                optimizer = optim.AdamW(
                    [
                        {"params": gain_or_bias_params, "weight_decay": 0.},
                        {"params": rest_params, "weight_decay": args.wd},
                    ],
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                )
            else:
                assert False, f'Unknown optimizer {opt}'

        if is_master(args):
            if is_master(args):
                defaults = copy.deepcopy(optimizer.defaults)
                defaults['weight_decay'] = args.wd
                defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
                logging.info(
                    f'Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}'
                )

        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = None
        if args.precision == "amp":
            try:
                scaler = torch.amp.GradScaler(device=device)
            except (AttributeError, TypeError) as e:
                scaler = torch.cuda.amp.GradScaler()
```

*   **描述:** 这段代码创建优化器（如 AdamW）和学习率调度器（如 cosine annealing）。优化器用于更新模型的权重，学习率调度器用于在训练过程中调整学习率。  AMP (自动混合精度) 使用 `torch.cuda.amp.GradScaler` ， 可以加速训练并减少内存占用.
*   **使用说明:**  根据需要选择合适的优化器和学习率调度器。 学习率调度器能够有效地控制训练过程。
*   **示例:**  命令行参数 `--lr` 和 `--wd` 用于设置学习率和权重衰减。

**7. 检查点加载和保存 (检查点加载与保存)**

```python
    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
```

```python
       # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)
```

*   **描述:** 这两段代码分别用于加载和保存检查点。加载检查点允许从之前的训练状态恢复，保存检查点可以在训练过程中定期保存模型的状态。
*   **使用说明:**  `--resume` 参数用于指定要加载的检查点文件。 `--save-frequency` 参数决定保存检查点的频率.  `--save_most_recent` 可以保存最新的模型到 `LATEST_CHECKPOINT_NAME` 定义的文件名。
*   **示例:**  使用 `--resume` 参数加载检查点：

    ```bash
    python your_script.py --resume /path/to/checkpoint.pt --train-data ...
    ```

**8. 训练和验证循环 (训练和验证循环)**

```python
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
        completed_epoch = epoch + 1

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            evaluate(model, data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer)
```

*   **描述:** 这是主要的训练循环。它迭代 `args.epochs` 轮，每轮调用 `train_one_epoch` 函数进行训练，并调用 `evaluate` 函数进行验证。
*   **使用说明:**  这个循环是训练的核心。 通过控制 `--epochs` 参数改变训练的总轮数。
*   **示例:**  无需手动调用，由 `main` 函数自动执行。