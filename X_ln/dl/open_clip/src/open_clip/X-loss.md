Lan: `py` From`X_ln\dl/open_clip/src\open_clip\loss.html`

好的，这次我会更注重代码质量和中文解释，并加入更清晰的示例。

**1. 优化 `gather_features` 函数**

```python
import torch
import torch.distributed as dist  # 确保导入
import torch.distributed.nn
from torch.nn import functional as F
from typing import Optional

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

def gather_features(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    local_loss: bool = False,
    gather_with_grad: bool = False,
    rank: int = 0,
    world_size: int = 1,
    use_horovod: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    在分布式训练中收集图像和文本特征。

    Args:
        image_features: 图像特征张量.
        text_features: 文本特征张量.
        local_loss: 是否计算本地损失.
        gather_with_grad: 是否收集带有梯度的特征.
        rank: 当前进程的排名.
        world_size: 总的进程数.
        use_horovod: 是否使用 Horovod.

    Returns:
        收集后的图像特征和文本特征张量。
    """
    if use_horovod:
        assert hvd is not None, "必须安装 Horovod 才能使用 Horovod 分布式训练."
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # 确保本地 rank 的梯度存在
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else: # 使用 torch.distributed
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)

            if not local_loss:
                # 确保本地 rank 的梯度存在
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features

# Demo 使用示例 (假设已初始化分布式环境)
if __name__ == '__main__':
  # 模拟分布式环境
  if torch.cuda.is_available():
      device = torch.device("cuda")
      print("CUDA is available. Using GPU.")
  else:
      device = torch.device("cpu")
      print("CUDA is not available. Using CPU.")

  torch.distributed.init_process_group(backend='gloo', init_method='tcp://localhost:12355', rank=0, world_size=1) # 使用 gloo backend, 适用于CPU
  rank = torch.distributed.get_rank()
  world_size = torch.distributed.get_world_size()

  # 创建一些模拟特征
  image_features = torch.randn(4, 64, device=device)  # 假设 batch_size=4, feature_dim=64
  text_features = torch.randn(4, 64, device=device)

  # 收集特征
  all_image_features, all_text_features = gather_features(
      image_features, text_features, local_loss=True, gather_with_grad=False, rank=rank, world_size=world_size
  )

  print(f"Rank {rank}: 收集后的图像特征形状: {all_image_features.shape}") # 期望形状: [4 * world_size, 64]
  print(f"Rank {rank}: 收集后的文本特征形状: {all_text_features.shape}") # 期望形状: [4 * world_size, 64]

  torch.distributed.destroy_process_group() # 清理分布式环境
```

**改进和解释:**

*   **类型提示 (Type Hints):**  添加了类型提示，使代码更易于阅读和理解。
*   **断言 (Assertion):**  添加了 `assert` 语句，以确保 Horovod 已正确安装。
*   **注释 (Comments):** 增加了更详细的注释，解释了代码的作用和参数。
*   **示例 (Example):** 提供了一个基本的使用示例，演示了如何使用 `gather_features` 函数。示例代码模拟了一个单进程的分布式环境，使用Gloo backend，更加通用。
*   **错误处理**: 补充CUDA不可用时的处理.
*   **torch.distributed的初始化和清理**: 补充了`init_process_group` 和 `destroy_process_group`， 更加完整.

**中文解释:**

这段代码的核心功能是在分布式训练中，将各个 GPU 上的图像和文本特征收集起来，以便计算全局的损失。`gather_features` 函数根据是否使用 Horovod 和是否需要梯度来选择不同的收集方法。`local_loss` 参数控制是否只使用本地 GPU 上的特征计算损失。

**2. 优化 `ClipLoss` 类**

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional

class ClipLoss(nn.Module):
    """
    计算 CLIP 损失。
    """

    def __init__(
        self,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
        use_horovod: bool = False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # 缓存状态
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device: torch.device, num_logits: int) -> torch.Tensor:
        """
        获取 ground truth 标签。
        """
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features: torch.Tensor, text_features: torch.Tensor, logit_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
      """获取logits."""
      if self.world_size > 1:
          all_image_features, all_text_features = gather_features(
              image_features,
              text_features,
              local_loss=self.local_loss,
              gather_with_grad=self.gather_with_grad,
              rank=self.rank,
              world_size=self.world_size,
              use_horovod=self.use_horovod,
          )

          if self.local_loss:
              logits_per_image = logit_scale * image_features @ all_text_features.T
              logits_per_text = logit_scale * text_features @ all_image_features.T
          else:
              logits_per_image = logit_scale * all_image_features @ all_text_features.T
              logits_per_text = logits_per_image.T
      else:
          logits_per_image = logit_scale * image_features @ text_features.T
          logits_per_text = logit_scale * text_features @ image_features.T
      return logits_per_image, logits_per_text

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        output_dict: bool = False,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        计算 CLIP 损失。
        """
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss

# Demo 使用示例
if __name__ == '__main__':
  if torch.cuda.is_available():
      device = torch.device("cuda")
      print("CUDA is available. Using GPU.")
  else:
      device = torch.device("cpu")
      print("CUDA is not available. Using CPU.")

  torch.distributed.init_process_group(backend='gloo', init_method='tcp://localhost:12355', rank=0, world_size=1)
  rank = torch.distributed.get_rank()
  world_size = torch.distributed.get_world_size()

  clip_loss = ClipLoss(local_loss=True, gather_with_grad=False, rank=rank, world_size=world_size)

  # 假设 image_features 和 text_features 已经从模型中提取
  image_features = torch.randn(4, 512, device=device)  # batch_size=4, feature_dim=512
  text_features = torch.randn(4, 512, device=device)
  logit_scale = torch.tensor(2.6592, device=device).exp()  # 假设 logit_scale 已经学习得到

  loss = clip_loss(image_features, text_features, logit_scale)
  print(f"Rank {rank}: CLIP 损失: {loss.item()}")

  output_dict = clip_loss(image_features, text_features, logit_scale, output_dict=True)
  print(f"Rank {rank}: CLIP 损失 (字典格式): {output_dict}")

  torch.distributed.destroy_process_group()
```

**改进和解释:**

*   **类型提示 (Type Hints):**  添加了类型提示，使代码更易于阅读和理解。
*   **文档字符串 (Docstrings):**  添加了文档字符串，解释了类的作用和方法。
*   **清晰的变量命名 (Clear Variable Names):** 使用更清晰的变量名，例如 `logits_per_image` 和 `logits_per_text`。
*   **明确返回值类型**:  使用`dict[str, torch.Tensor] | torch.Tensor`更精确地定义返回值类型.
*   **更明确的get_logits函数**: 分离出了`get_logits`函数.

**中文解释:**

`ClipLoss` 类实现了 CLIP 模型的损失函数。它接收图像和文本特征，并计算它们之间的对比损失。`get_ground_truth` 方法生成 ground truth 标签，用于计算交叉熵损失。`forward` 方法计算损失，并返回一个包含 `contrastive_loss` 的字典（如果 `output_dict` 为 `True`）。

**3. 优化 `CoCaLoss` 类**

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional

class CoCaLoss(nn.Module):
    """
    计算 CoCa 损失。
    """

    def __init__(
        self,
        caption_loss_weight: float,
        clip_loss_weight: float,
        pad_id: int = 0,  # pad_token for open_clip custom tokenizer
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
        use_horovod: bool = False,
    ):
        super().__init__()
        self.clip_loss = ClipLoss(local_loss, gather_with_grad, cache_labels, rank, world_size, use_horovod)
        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        logit_scale: torch.Tensor,
        output_dict: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        """
        计算 CoCa 损失。
        """
        if self.clip_loss_weight:
            clip_loss = self.clip_loss(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss
        else:
            clip_loss = torch.tensor(0, device=logits.device)

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),  # logits: [B, seq_len, vocab_size]
            labels,  # labels: [B, seq_len]
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


# Demo 使用示例
if __name__ == '__main__':
  if torch.cuda.is_available():
      device = torch.device("cuda")
      print("CUDA is available. Using GPU.")
  else:
      device = torch.device("cpu")
      print("CUDA is not available. Using CPU.")

  torch.distributed.init_process_group(backend='gloo', init_method='tcp://localhost:12355', rank=0, world_size=1)
  rank = torch.distributed.get_rank()
  world_size = torch.distributed.get_world_size()

  coca_loss = CoCaLoss(
      caption_loss_weight=0.5,
      clip_loss_weight=0.5,
      pad_id=0,
      local_loss=True,
      gather_with_grad=False,
      rank=rank,
      world_size=world_size,
  )

  # 假设 image_features, text_features, logits, labels 已经从模型中提取
  image_features = torch.randn(4, 512, device=device)
  text_features = torch.randn(4, 512, device=device)
  logits = torch.randn(4, 32, 1000, device=device)  # batch_size=4, seq_len=32, vocab_size=1000
  labels = torch.randint(0, 1000, (4, 32), device=device)
  logit_scale = torch.tensor(2.6592, device=device).exp()

  clip_loss, caption_loss = coca_loss(image_features, text_features, logits, labels, logit_scale)
  print(f"Rank {rank}: CLIP 损失: {clip_loss.item()}")
  print(f"Rank {rank}: Caption 损失: {caption_loss.item()}")

  output_dict = coca_loss(image_features, text_features, logits, labels, logit_scale, output_dict=True)
  print(f"Rank {rank}: CoCa 损失 (字典格式): {output_dict}")

  torch.distributed.destroy_process_group()
```

**改进和解释:**

*   **更清晰的初始化**: 将 `ClipLoss` 作为成员初始化，避免代码重复.
*   **类型提示 (Type Hints):**  添加了类型提示，使代码更易于阅读和理解。
*   **文档字符串 (Docstrings):**  添加了文档字符串，解释了类的作用和方法。
*   **参数命名 (Parameter Names):**  使用了更有意义的参数名，例如 `caption_loss_weight` 和 `clip_loss_weight`。
*   **损失权重 (Loss Weights):** 使用 `clip_loss_weight` 和 `caption_loss_weight` 控制损失的权重。

**中文解释:**

`CoCaLoss` 类实现了 CoCa 模型的损失函数。它结合了 CLIP 损失和 captioning 损失。CLIP 损失衡量图像和文本特征之间的相似性，而 captioning 损失衡量模型生成文本描述的准确性。 `forward` 方法计算总损失，并可以选择返回包含各个损失的字典。

**4. 优化 `DistillClipLoss` 类**

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional

class DistillClipLoss(nn.Module):
    """
    计算 DistillClip 损失。
    """

    def __init__(
        self,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
        use_horovod: bool = False,
    ):
        super().__init__()
        self.clip_loss = ClipLoss(local_loss, gather_with_grad, cache_labels, rank, world_size, use_horovod)

    def dist_loss(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
        """
        计算知识蒸馏损失。
        """
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        dist_image_features: torch.Tensor,
        dist_text_features: torch.Tensor,
        dist_logit_scale: torch.Tensor,
        output_dict: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        """
        计算 DistillClip 损失。
        """
        logits_per_image, logits_per_text = self.clip_loss.get_logits(image_features, text_features, logit_scale)
        dist_logits_per_image, dist_logits_per_text = self.clip_loss.get_logits(dist_image_features, dist_text_features, dist_logit_scale)
        device = image_features.device
        labels = self.clip_loss.get_ground_truth(device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image)
            + self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


# Demo 使用示例
if __name__ == '__main__':
  if torch.cuda.is_available():
      device = torch.device("cuda")
      print("CUDA is available. Using GPU.")
  else:
      device = torch.device("cpu")
      print("CUDA is not available. Using CPU.")

  torch.distributed.init_process_group(backend='gloo', init_method='tcp://localhost:12355', rank=0, world_size=1)
  rank = torch.distributed.get_rank()
  world_size = torch.distributed.get_world_size()

  distill_clip_loss = DistillClipLoss(local_loss=True, gather_with_grad=False, rank=rank, world_size=world_size)

  # 假设 image_features, text_features, dist_image_features, dist_text_features, logit_scale, dist_logit_scale 已经从模型中提取
  image_features = torch.randn(4, 512, device=device)
  text_features = torch.randn(4, 512, device=device)
  dist_image_features = torch.randn(4, 512, device=device)
  dist_text_features = torch.randn(4, 512, device=device)
  logit_scale = torch.tensor(2.6592, device=device).exp()
  dist_logit_scale = torch.tensor(2.6592, device=device).exp()

  contrastive_loss, distill_loss = distill_clip_loss(
      image_features, text_features, logit_scale, dist_image_features, dist_text_features, dist_logit_scale
  )
  print(f"Rank {rank}: Contrastive 损失: {contrastive_loss.item()}")
  print(f"Rank {rank}: Distill 损失: {distill_loss.item()}")

  output_dict = distill_clip_loss(
      image_features, text_features, logit_scale, dist_image_features, dist_text_features, dist_logit_scale, output_dict=True
  )
  print(f"Rank {rank}: DistillClip 损失 (字典格式): {output_dict}")

  torch.distributed.destroy_process_group()
```

**改进和解释:**

*   **继承和组合**: 继承 `nn.Module` 并组合 `ClipLoss` 以避免代码重复.  `ClipLoss` 的功能可以被复用，只需在 `DistillClipLoss` 中使用它。
*   **更清晰的损失计算**: 使用 `self.clip_loss.get_logits` 获取logits， 更清晰.
*    **类型提示 (Type Hints):**  添加了类型提示，使代码更易于阅读和理解。
*   **文档字符串 (Docstrings):**  添加了文档字符串，解释了类的作用和方法。

**中文解释:**

`DistillClipLoss` 类实现了知识蒸馏版本的 CLIP 损失函数。它使用一个“教师”模型（`dist_image_features` 和 `dist_text_features`）来指导“学生”模型（`image_features` 和 `text_features`）的学习。除了标准的 CLIP 损失之外，还计算了一个蒸馏损失，用于衡量学生模型 logits 与教师模型 logits 的相似性。

**5. 优化 `SigLipLoss` 类**

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional

class SigLipLoss(nn.Module):
    """Sigmoid Loss for Language Image Pre-Training (SigLIP)"""

    def __init__(
        self,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
        dist_impl: Optional[str] = None,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl or "bidir"  # 默认为 bidir
        assert self.dist_impl in ("bidir", "shift", "reduce", "gather")

        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(
        self, device: torch.device, dtype: torch.dtype, num_logits: int, negative_only: bool = False
    ) -> torch.Tensor:
        """生成 ground truth 标签."""
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算 logits."""
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: Optional[torch.Tensor] = None,
        negative_only: bool = False,
    ) -> torch.Tensor:
        """计算 SigLIP 损失."""
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device, image_features.dtype, image_features.shape[0], negative_only=negative_only
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: Optional[torch.Tensor],
        output_dict: bool = False,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """计算 SigLIP 损失."""
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            if self.dist_impl == "bidir":
                from neighbour_exchange import neighbour_exchange_bidir_with_grad, neighbour_exchange_with_grad

                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank, right_rank, text_features_to_left, text_features_to_right
                    )
                    for f in text_features_recv:
                        loss += self._loss(image_features, f, logit_scale, logit_bias, negative_only=True)
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(left_rank, right_rank, text_features_to_right)
                    loss += self._loss(image_features, text_features_recv, logit_scale, logit_bias, negative_only=True)
            elif self.dist_impl == "shift":
                from neighbour_exchange import neighbour_exchange_with_grad

                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(left_rank, right_rank, text_features_to_right)
                    loss += self._loss(image_features, text_features_from_left, logit_scale, logit_bias, negative_only=True)
                    text_features_to_right = text_features_from_left
            elif self.dist_impl == "reduce":
                import torch.distributed.nn

                for i in range(self.world_size):
                    text_from_other = torch.distributed.nn.all_reduce(
                        text_features * (self.rank == i), torch.distributed.ReduceOp.SUM
                    )
                    loss += float(i != self.rank) * self._loss(
                        image_features, text_from_other, logit_scale, logit_bias, negative_only=True
                    )
            elif self.dist_impl == "gather":
                import torch.distributed.nn

                all_text = torch.distributed.nn.all_gather(text_features)
                for i in range(self.world_size):
                    loss += float(i != self.rank) * self._loss(
                        image_features, all_text[i], logit_scale, logit_bias, negative_only=True
                    )
            else:
                assert False

        return {"contrastive_loss": loss} if output_dict else loss


# Create a file named `neighbour_exchange.py` with the functions
# neighbour_exchange, neighbour_exchange_bidir,
# neighbour_exchange_with_grad, neighbour_exchange_bidir_with_grad
# These functions need to be defined based on the original implementation

# Demo Usage
if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # Mock neighbour_exchange functions for single process testing
    def mock_neighbour_exchange_with_grad(from_rank, to_rank, tensor):
        return tensor  # In single process, just return the tensor itself

    def mock_neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right):
        return tensor_to_right, tensor_to_left  # Return swapped tensors

    # Register mock functions
    import sys
    sys.modules['neighbour_exchange'] = type(sys)('neighbour_exchange')
    sys.modules['neighbour_exchange'].neighbour_exchange_with_grad = mock_neighbour_exchange_with_grad
    sys.modules['neighbour_exchange'].neighbour_exchange_bidir_with_grad = mock_neighbour_exchange_bidir_with_grad

    torch.distributed.init_process_group(backend='gloo', init_method='tcp://localhost:12355', rank=0, world_size=1)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    siglip_loss = SigLipLoss(rank=rank, world_size=world_size, dist_impl="bidir")

    # 假设 image_features, text_features, logit_scale, logit_bias 已经从模型中提取
    image_features = torch.randn(4, 512, device=device)
    text_features = torch.randn(4, 512, device=device)
    logit_scale = torch.tensor(2.6592, device=device).exp()
    logit_bias = torch.randn(512, 512, device=device)  # 假设 logit_bias 存在

    loss = siglip_loss(image_features, text_features, logit_scale, logit_bias)
    print(f"Rank {rank}: SigLIP 损失: {loss.item()}")

    output_dict = siglip_loss(image_features, text_features, logit_scale, logit_bias, output_dict=True)
    print(f"Rank {rank}: SigLIP 损失 (字典格式): {output_dict}")

    torch.distributed.destroy_process_group()

```

**改进和解释:**

*   **类型提示 (Type Hints):**  添加了类型提示，使代码更易于阅读和理解。
*   **文档字符串 (Docstrings):**  添加了文档字符串，解释了类的作用和方法。
*   **默认 `dist_impl`:**  为 `dist_impl` 参数提供了一个默认值 (`"bidir"`).
*   **更清晰的损失计算**: 进行了代码重构，使损失计算更加清晰.
*   **导入语句 (Import Statements):**  将导入语句放在条件语句中，避免了不必要的依赖。
*   **创建`neighbour_exchange.py`的提示**: 增加提示，需要用户创建`neighbour_exchange.py`， 否则代码无法执行.
*    **单进程测试mock**: 为了能在单进程中执行，增加了mock函数。

**中文解释:**

`SigLipLoss` 类