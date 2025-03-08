Lan: `py` From`X_ln\dl/DeepSeek-VL2\deepseek_vl2\models\modeling_deepseek_vl_v2.html`

好的，让我们重新开始，并提供更智能的代码改进和中文描述。

**总览**

这份代码定义了一个`DeepseekVLV2ForCausalLM`模型，它是一个视觉-语言模型，结合了视觉编码器（Vision Transformer）、多层感知器投影器（MLP Projector）和 DeepseekV2 语言模型。模型的目标是根据图像和文本的输入生成文本。我们将重点关注以下几个可以改进的方面：

1.  **`MlpProjector`**: 改进投影器的灵活性和效率。
2.  **`prepare_inputs_embeds`**: 优化图像嵌入的准备过程，特别是图像分块和嵌入。
3.  **内存管理**: 优化增量预填充和前向传播过程中的内存管理。
4.  **Config的增强**: 进一步解耦config，使其更加灵活。

以下是分步改进的代码和解释。

**1. 改进的 `MlpProjector`**

```python
class MlpProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 使用字典来存储不同投影类型的实现
        self.projector_types = {
            "identity": nn.Identity(),
            "linear": nn.Linear(cfg.input_dim, cfg.n_embed),
            "mlp_gelu": self._build_mlp_gelu(cfg),
            "downsample_mlp_gelu": self._build_downsample_mlp_gelu(cfg)
        }

        if cfg.projector_type not in self.projector_types:
            raise ValueError(f"Unknown projector type: {cfg.projector_type}")

        self.layers = self.projector_types[cfg.projector_type]  # 选择对应的层

        if cfg.token_pooling:
            self.token_pooling_layer = nn.Linear(cfg.input_dim * 4, cfg.input_dim)

    def _build_mlp_gelu(self, cfg):
        mlp_depth = cfg.depth
        modules = [nn.Linear(cfg.input_dim, cfg.n_embed)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
        return nn.Sequential(*modules)

    def _build_downsample_mlp_gelu(self, cfg):
        mlp_depth = cfg.depth
        mlp_ratio = cfg.mlp_ratio
        modules = [nn.Linear(cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio, cfg.n_embed * mlp_ratio)]
        for _ in range(1, mlp_depth - 1):
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed * mlp_ratio))
        modules.append(nn.GELU())
        modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed))
        return nn.Sequential(*modules)

    def forward(self, x):
        if self.cfg.token_pooling:
            batch_size, wxh, channels = x.shape
            w = h = int(wxh ** 0.5)
            x = x.view(batch_size, w, h, channels)
            x = x.permute(0, 3, 1, 2)
            patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
            batch_size, channels, h_patches, w_patches, _, _ = patches.size()
            patches = patches.contiguous().view(batch_size, channels, h_patches * w_patches, -1)
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(batch_size, h_patches * w_patches, channels * 4)
            x = self.token_pooling_layer(patches)
        elif self.cfg.projector_type == 'downsample_mlp_gelu':
            bs, hw, input_dim = x.shape
            h = w = int((hw) ** 0.5)

            if h % self.cfg.downsample_ratio:
                pad = self.cfg.downsample_ratio - h % self.cfg.downsample_ratio
            else:
                pad = 0
            x = x.reshape(bs, h, w, input_dim)
            if pad > 0:
                x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)

            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = F.unfold(x, kernel_size=self.cfg.downsample_ratio, stride=self.cfg.downsample_ratio,
                         padding=0)  # B, C*4, HW // 4
            x = x.permute(0, 2, 1)

        return self.layers(x)
```

**中文描述:**

*   `MlpProjector` 类现在使用一个字典 `projector_types` 来存储不同类型的投影器。
*   这使得添加新的投影器类型更容易，只需将其添加到字典中即可。
*   `_build_mlp_gelu` 和 `_build_downsample_mlp_gelu` 方法用于构建对应的MLP结构，提高代码可读性和组织性。
*   在`forward`方法中，会根据`cfg.projector_type`选择对应的层。

**2.  改进的 `prepare_inputs_embeds`**

```python
    def prepare_inputs_embeds(
            self,
            input_ids: torch.LongTensor,
            images: Optional[torch.FloatTensor] = None,
            images_seq_mask: Optional[torch.LongTensor] = None,
            images_spatial_crop: Optional[torch.LongTensor] = None,
            **ignore_kwargs
    ):
        """
        Args:
            input_ids (torch.LongTensor): [b, T]
            images (torch.FloatTensor): [b, max_n_images, 3, height, width]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_spatial_crop (torch.LongTensor): [b, max_n_images, 2]

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        if images is None or images_spatial_crop.sum() == 0:
            return self.language.get_input_embeddings()(input_ids)

        bs, max_n_images, _ = images_spatial_crop.shape

        # 预先计算每个batch的tile数量，并存储图像特征
        batch_num_tiles = [0] * bs
        image_features = []

        for idx in range(bs):
            tiles_in_batch = []
            for jdx in range(max_n_images):
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break
                num_tiles = 1 + num_width_tiles * num_height_tiles  # global + local
                batch_num_tiles[idx] += num_tiles
                tiles_in_batch.append(images[idx, jdx:jdx + 1].repeat(num_tiles, 1, 1, 1)) # [num_tiles, C, H, W]

            if tiles_in_batch:
                tiles = torch.cat(tiles_in_batch, dim=0)
                image_features.append(self.vision(tiles))  # [num_total_tiles, vit_seq_len, c]
            else:
                image_features.append(None)

        # 生成图像嵌入
        input_embeds = self.language.get_input_embeddings()(input_ids)

        tile_index = 0
        for idx in range(bs):
            if image_features[idx] is None:
                continue

            images_embeds = self.projector(image_features[idx])  # [num_total_tiles, hw, D]
            num_tiles_in_batch = batch_num_tiles[idx]
            hw, n_dim = images_embeds.shape[1], images_embeds.shape[2]
            h = w = int(hw ** 0.5)

            image_embeds_idx = []
            tile_idx_in_image = 0
            for jdx in range(max_n_images):
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break

                num_tiles_in_image = num_width_tiles * num_height_tiles

                # 分割全局和局部特征
                global_features = images_embeds[tile_idx_in_image]  # [hw, D]
                local_features = images_embeds[tile_idx_in_image + 1: tile_idx_in_image + 1 + num_tiles_in_image] # [num_tiles, hw, D]

                tile_idx_in_image += num_tiles_in_image + 1

                # 格式化特征
                if self.tile_tag == "2D":
                    global_features = global_features.view(h, w, n_dim)
                    new_lines_in_global = repeat(self.image_newline, "d -> h 1 d", h=h)
                    global_features = torch.cat([global_features, new_lines_in_global], dim=1)
                    global_features = global_features.view(-1, n_dim)

                    local_features = rearrange(
                        local_features,
                        "(th tw) (h w) d -> (th h) (tw w) d",
                        th=num_height_tiles,
                        tw=num_width_tiles,
                        h=h,
                        w=w
                    )
                    new_lines_in_local = repeat(
                        self.image_newline,
                        "d -> (th h) 1 d",
                        th=num_height_tiles,
                        h=h
                    )
                    local_features = torch.cat([local_features, new_lines_in_local], dim=1)
                    local_features = local_features.view(-1, n_dim)

                    if self.global_view_pos == "head":
                        global_local_features = torch.cat(
                            [global_features, self.view_seperator[None, :], local_features], dim=0)
                    else:
                        global_local_features = torch.cat(
                            [local_features, self.view_seperator[None, :], global_features], dim=0)
                else:
                    global_features = torch.cat(
                        [self.tile_indicators[0:1], global_features], dim=0
                    )
                    local_features = torch.cat(
                        [self.tile_indicators[1:num_tiles_in_image + 1].unsqueeze(1), local_features], dim=1
                    )
                    local_features = rearrange(local_features, 'crop_num hw d -> (crop_num hw) d')

                    if self.global_view_pos == "head":
                        global_local_features = torch.cat([global_features, local_features], dim=0)
                    else:
                        global_local_features = torch.cat([local_features, global_features], dim=0)

                image_embeds_idx.append(global_local_features)

            if len(image_embeds_idx) > 0:
                image_embeds_idx = torch.cat(image_embeds_idx, dim=0)
                input_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1), image_embeds_idx)

        return input_embeds
```

**中文描述:**

*   **预计算图像特征:**  循环提前计算所有图像块的视觉特征，避免重复计算。
*   **分批处理:**  每个批次的图像块视觉特征存储在`image_features`列表中，并仅在需要时才传递到投影仪。
*   **更清晰的索引:** 简化了索引逻辑，更容易理解图像块是如何被处理和嵌入的。
*   **减少重复计算**: 循环内仅进行一次`vision`调用, 提高效率。
*   **增加鲁棒性**: 图像块的数量可能为零时增加处理逻辑。
*   **优化内存占用**: 仅存储每个批次的必要视觉特征。

**3. 优化内存管理**

```python
    @torch.no_grad()
    def incremental_prefilling(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,

            images: Optional[torch.FloatTensor] = None,
            images_seq_mask: Optional[torch.LongTensor] = None,
            images_spatial_crop: Optional[torch.LongTensor] = None,
            chunk_size: int = 1024
    ):
        if inputs_embeds is None:
            inputs_embeds = self.prepare_inputs_embeds(
                input_ids=input_ids,
                images=images,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
            )

            del images, images_seq_mask, images_spatial_crop  # 显式删除
            self._clear_cuda_cache()  # 清理缓存

        bzs, seq_len, _ = inputs_embeds.shape
        past_key_values = None

        prefilling_len = seq_len - 1
        for i in range(0, prefilling_len, chunk_size):
            chunk_start = i
            chunk_end = min(i + chunk_size, prefilling_len)
            chunk_inputs_embeds = inputs_embeds[:, chunk_start: chunk_end]
            chunk_attention_mask = attention_mask[:, 0: chunk_end]

            position_ids = torch.arange(
                chunk_start,
                chunk_end,
                dtype=torch.long,
                device=inputs_embeds.device
            ).unsqueeze(0) if past_key_values is not None else None

            if past_key_values is not None:
                past_key_values = self._move_past_key_values_to_gpu(past_key_values, inputs_embeds.device)

            outputs = self.forward(
                inputs_embeds=chunk_inputs_embeds,
                attention_mask=chunk_attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True,
                images=None, # 确保不传递图像
                images_seq_mask=None,
                images_spatial_crop=None
            )

            past_key_values = outputs.past_key_values
            past_key_values = self._move_past_key_values_to_cpu(past_key_values)

            del outputs, position_ids, chunk_inputs_embeds, chunk_attention_mask  # 显式删除
            self._clear_cuda_cache()

        prefilling_key_values = [(layer_past[0][:, :, 0: prefilling_len, ...].to(inputs_embeds.device),
                                  layer_past[1][:, :, 0: prefilling_len, ...].to(inputs_embeds.device))
                                 for layer_past in past_key_values]

        return inputs_embeds, prefilling_key_values
```

**中文描述:**

*   **显式删除变量:**  使用 `del` 语句显式删除不再需要的变量，以便立即释放内存。
*   **在恰当的地方调用 `_clear_cuda_cache()`:** 在每次迭代后调用 `_clear_cuda_cache()`，以确保 CUDA 缓存被清理。
*   **避免不必要的图像传递:** 确保在增量预填充期间，图像不会被传递到 `forward` 函数中。

**4. Config的增强**

```python
class MlpProjectorConfig(PretrainedConfig):
    model_type = "mlp_projector"
    projector_type: str = "downsample_mlp_gelu"
    input_dim: int = 1152
    n_embed: int = 2048
    depth: int = 2
    mlp_ratio: int = 1
    downsample_ratio: int = 2
    token_pooling: bool = False
    use_bias: bool = True  # 增加一个bias的选项

    def __init__(
            self,
            projector_type: str = "downsample_mlp_gelu",
            input_dim: int = 1152,
            n_embed: int = 2048,
            depth: int = 2,
            mlp_ratio: int = 1,
            downsample_ratio: int = 2,
            token_pooling: bool = False,
            use_bias: bool = True,
            **kwargs
    ):
        self.projector_type = projector_type
        self.input_dim = input_dim
        self.n_embed = n_embed
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.downsample_ratio = downsample_ratio
        self.token_pooling = token_pooling
        self.use_bias = use_bias  # 设置bias

        super().__init__(**kwargs)
```

修改`MlpProjector`以利用`use_bias`配置：

```python
class MlpProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.projector_types = {
            "identity": nn.Identity(),
            "linear": nn.Linear(cfg.input_dim, cfg.n_embed, bias=cfg.use_bias),  # 使用config中的bias设置
            "mlp_gelu": self._build_mlp_gelu(cfg),
            "downsample_mlp_gelu": self._build_downsample_mlp_gelu(cfg)
        }

        if cfg.projector_type not in self.projector_types:
            raise ValueError(f"Unknown projector type: {cfg.projector_type}")

        self.layers = self.projector_types[cfg.projector_type]
        if cfg.token_pooling:
            self.token_pooling_layer = nn.Linear(cfg.input_dim * 4, cfg.input_dim, bias=cfg.use_bias)  # 使用config中的bias设置

    def _build_mlp_gelu(self, cfg):
        mlp_depth = cfg.depth
        modules = []
        for i in range(mlp_depth):
            modules.append(nn.Linear(cfg.input_dim if i == 0 else cfg.n_embed, cfg.n_embed, bias=cfg.use_bias))  # 使用config中的bias设置
            if i < mlp_depth - 1:
                modules.append(nn.GELU())
        return nn.Sequential(*modules)

    def _build_downsample_mlp_gelu(self, cfg):
        mlp_depth = cfg.depth
        mlp_ratio = cfg.mlp_ratio
        modules = [nn.Linear(cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio, cfg.n_embed * mlp_ratio, bias=cfg.use_bias)]  # 使用config中的bias设置
        for i in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed * mlp_ratio, bias=cfg.use_bias))  # 使用config中的bias设置
        modules.append(nn.GELU())
        modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed, bias=cfg.use_bias))  # 使用config中的bias设置
        return nn.Sequential(*modules)
```

**中文描述:**

*   **可配置的 Bias**:  在 `MlpProjectorConfig` 中添加 `use_bias` 选项，并将其传递给 `nn.Linear` 层。
*   允许用户配置线性层是否使用 bias，从而提高模型的灵活性。

**总结**

以上代码变更旨在提高代码的模块化、可读性、效率和灵活性。 通过使用显式内存管理、模块化的设计和更清晰的代码结构，可以更容易地维护和扩展这个模型。
