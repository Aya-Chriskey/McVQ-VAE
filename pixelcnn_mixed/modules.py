import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if 'Conv' in classname or 'Linear' in classname:
        try:
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        except AttributeError:
            pass
    elif 'Embedding' in classname:
        try:
            nn.init.uniform_(module.weight.data, -1.0 / max(1, module.num_embeddings), 1.0 / max(1, module.num_embeddings))
        except AttributeError:
            pass


class GatedActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_tanh, x_sigmoid = x.chunk(2, dim=1)
        return torch.tanh(x_tanh) * torch.sigmoid(x_sigmoid)


class ConditionalGatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type: str, dim: int, kernel: int, residual: bool = True, n_classes: int = 10, cond_dim: int = None):
        super().__init__()
        assert mask_type in {'A', 'B'}
        assert kernel % 2 == 1
        self.mask_type = mask_type
        self.residual = residual

        self.class_cond_embedding = nn.Embedding(n_classes, 2 * dim)
        self.cond_proj = nn.Conv2d(cond_dim, 2 * dim, 1) if cond_dim is not None else None

        kernel_shp = (kernel // 2 + 1, kernel)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(dim, dim * 2, kernel_shp, 1, padding_shp)
        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(dim, dim * 2, kernel_shp, 1, padding_shp)
        self.horiz_resid = nn.Conv2d(dim, dim, 1)
        self.gate = GatedActivation()

    def make_causal(self) -> None:
        self.vert_stack.weight.data[:, :, -1].zero_()
        self.horiz_stack.weight.data[:, :, :, -1].zero_()

    def forward(self, x_v: torch.Tensor, x_h: torch.Tensor, labels: torch.Tensor, cond: torch.Tensor = None):
        if self.mask_type == 'A':
            self.make_causal()

        label_term = self.class_cond_embedding(labels)[:, :, None, None]
        cond_term = 0
        if self.cond_proj is not None and cond is not None:
            cond_term = self.cond_proj(cond)

        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-2), :x_v.size(-1)]
        out_v = self.gate(h_vert + label_term + cond_term)

        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-1)]
        v2h = self.vert_to_horiz(h_vert)
        out = self.gate(v2h + h_horiz + label_term + cond_term)

        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)
        return out_v, out_h


class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim: int, dim: int = 64, n_layers: int = 15, n_classes: int = 10):
        super().__init__()
        self.input_dim = int(input_dim)
        self.dim = int(dim)
        self.n_classes = int(n_classes)
        self.embedding = nn.Embedding(self.input_dim, self.dim)
        self.layers = nn.ModuleList()
        for idx in range(n_layers):
            mask_type = 'A' if idx == 0 else 'B'
            kernel = 7 if idx == 0 else 3
            residual = idx != 0
            self.layers.append(
                ConditionalGatedMaskedConv2d(mask_type, self.dim, kernel, residual, n_classes=self.n_classes, cond_dim=None)
            )
        self.output_conv = nn.Sequential(
            nn.Conv2d(self.dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, self.input_dim, 1),
        )
        self.apply(weights_init)

    def _embed_indices(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.size() + (-1,)
        x = self.embedding(x.reshape(-1)).view(shp)
        return x.permute(0, 3, 1, 2).contiguous()

    def build_condition_feature(self, prev_branch_maps: torch.Tensor = None) -> torch.Tensor:
        del prev_branch_maps
        return None

    def forward(self, x: torch.Tensor, labels: torch.Tensor, prev_branch_maps: torch.Tensor = None) -> torch.Tensor:
        x = self._embed_indices(x)
        cond = self.build_condition_feature(prev_branch_maps)
        x_v, x_h = x, x
        for layer in self.layers:
            x_v, x_h = layer(x_v, x_h, labels, cond)
        return self.output_conv(x_h)

    @torch.no_grad()
    def generate(self, labels: torch.Tensor, shape, batch_size: int = None, prev_branch_maps: torch.Tensor = None, device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        labels = labels.to(device)
        if batch_size is None:
            batch_size = int(labels.size(0))
        assert batch_size == int(labels.size(0)), 'batch_size must equal len(labels) for class-conditional generation.'
        height, width = int(shape[0]), int(shape[1])
        samples = torch.zeros((batch_size, height, width), dtype=torch.long, device=device)
        for i in range(height):
            for j in range(width):
                logits = self.forward(samples, labels, prev_branch_maps=prev_branch_maps)
                probs = F.softmax(logits[:, :, i, j], dim=-1)
                samples[:, i, j] = torch.multinomial(probs, 1).squeeze(-1)
        return samples


class ConditionalGatedPixelCNN(GatedPixelCNN):
    def __init__(self, input_dim: int, dim: int = 64, n_layers: int = 15, n_classes: int = 10, cond_input_dims=None):
        self.cond_input_dims = [int(x) for x in (cond_input_dims or [])]
        super().__init__(input_dim=input_dim, dim=dim, n_layers=n_layers, n_classes=n_classes)
        if len(self.cond_input_dims) == 0:
            raise ValueError('ConditionalGatedPixelCNN requires at least one previous-branch codebook size.')
        self.condition_embeddings = nn.ModuleList([nn.Embedding(k, self.dim) for k in self.cond_input_dims])
        self.condition_projection = nn.Conv2d(len(self.cond_input_dims) * self.dim, self.dim, 1)
        self.layers = nn.ModuleList()
        for idx in range(n_layers):
            mask_type = 'A' if idx == 0 else 'B'
            kernel = 7 if idx == 0 else 3
            residual = idx != 0
            self.layers.append(
                ConditionalGatedMaskedConv2d(mask_type, self.dim, kernel, residual, n_classes=self.n_classes, cond_dim=self.dim)
            )
        self.condition_embeddings.apply(weights_init)
        self.condition_projection.apply(weights_init)
        self.output_conv.apply(weights_init)

    def build_condition_feature(self, prev_branch_maps: torch.Tensor = None) -> torch.Tensor:
        if prev_branch_maps is None:
            raise ValueError('ConditionalGatedPixelCNN requires prev_branch_maps with shape [B, C, H, W].')
        if prev_branch_maps.dim() != 4:
            raise ValueError(f'prev_branch_maps must have shape [B, C, H, W], got {tuple(prev_branch_maps.shape)}')
        if prev_branch_maps.size(1) != len(self.cond_input_dims):
            raise ValueError(f'Expected {len(self.cond_input_dims)} previous branches, got {prev_branch_maps.size(1)}')
        feats = []
        batch_size, _, height, width = prev_branch_maps.shape
        for branch_idx, emb in enumerate(self.condition_embeddings):
            branch = prev_branch_maps[:, branch_idx, :, :].reshape(-1)
            feat = emb(branch).view(batch_size, height, width, self.dim).permute(0, 3, 1, 2).contiguous()
            feats.append(feat)
        cond = torch.cat(feats, dim=1)
        return self.condition_projection(cond)


class IndependentMixedPixelCNNPrior(nn.Module):
    def __init__(self, codebook_sizes, hidden_size: int = 64, n_layers: int = 15, n_classes: int = 10):
        super().__init__()
        self.codebook_sizes = [int(k) for k in codebook_sizes]
        self.num_branches = len(self.codebook_sizes)
        self.priors = nn.ModuleList([
            GatedPixelCNN(input_dim=k, dim=hidden_size, n_layers=n_layers, n_classes=n_classes)
            for k in self.codebook_sizes
        ])

    def forward(self, indices: torch.Tensor, labels: torch.Tensor):
        if indices.dim() != 4 or indices.size(1) != self.num_branches:
            raise ValueError(f'indices must have shape [B, {self.num_branches}, H, W], got {tuple(indices.shape)}')
        return [self.priors[idx](indices[:, idx, :, :], labels) for idx in range(self.num_branches)]

    def loss(self, indices: torch.Tensor, labels: torch.Tensor):
        logits_per_branch = self.forward(indices, labels)
        losses = []
        for idx, logits in enumerate(logits_per_branch):
            target = indices[:, idx, :, :]
            losses.append(F.cross_entropy(logits, target))
        mean_loss = torch.stack(losses).mean()
        return mean_loss, logits_per_branch, losses

    @torch.no_grad()
    def generate(self, labels: torch.Tensor, shape, device: torch.device = None) -> torch.Tensor:
        samples = [prior.generate(labels=labels, shape=shape, batch_size=int(labels.size(0)), device=device) for prior in self.priors]
        return torch.stack(samples, dim=1)


class ConditionalMixedPixelCNNPrior(nn.Module):
    def __init__(self, codebook_sizes, hidden_size: int = 64, n_layers: int = 15, n_classes: int = 10):
        super().__init__()
        self.codebook_sizes = [int(k) for k in codebook_sizes]
        self.num_branches = len(self.codebook_sizes)
        if self.num_branches <= 0:
            raise ValueError('num_branches must be positive.')
        priors = [GatedPixelCNN(input_dim=self.codebook_sizes[0], dim=hidden_size, n_layers=n_layers, n_classes=n_classes)]
        for idx in range(1, self.num_branches):
            priors.append(
                ConditionalGatedPixelCNN(
                    input_dim=self.codebook_sizes[idx],
                    dim=hidden_size,
                    n_layers=n_layers,
                    n_classes=n_classes,
                    cond_input_dims=self.codebook_sizes[:idx],
                )
            )
        self.priors = nn.ModuleList(priors)

    def forward(self, indices: torch.Tensor, labels: torch.Tensor):
        if indices.dim() != 4 or indices.size(1) != self.num_branches:
            raise ValueError(f'indices must have shape [B, {self.num_branches}, H, W], got {tuple(indices.shape)}')
        outputs = []
        for idx, prior in enumerate(self.priors):
            prev_maps = None if idx == 0 else indices[:, :idx, :, :]
            outputs.append(prior(indices[:, idx, :, :], labels, prev_branch_maps=prev_maps))
        return outputs

    def loss(self, indices: torch.Tensor, labels: torch.Tensor):
        logits_per_branch = self.forward(indices, labels)
        losses = []
        for idx, logits in enumerate(logits_per_branch):
            target = indices[:, idx, :, :]
            losses.append(F.cross_entropy(logits, target))
        mean_loss = torch.stack(losses).mean()
        return mean_loss, logits_per_branch, losses

    @torch.no_grad()
    def generate(self, labels: torch.Tensor, shape, device: torch.device = None) -> torch.Tensor:
        generated = []
        for idx, prior in enumerate(self.priors):
            prev_maps = None if idx == 0 else torch.stack(generated, dim=1)
            current = prior.generate(labels=labels, shape=shape, batch_size=int(labels.size(0)), prev_branch_maps=prev_maps, device=device)
            generated.append(current)
        return torch.stack(generated, dim=1)
