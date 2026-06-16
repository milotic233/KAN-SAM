
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

__all__ = [
    "wbce_loss",
    "soft_dice_loss",
    "iou_loss",
    "ssim_loss",
    "boundary_bce_loss",
    "soft_fbeta_loss",
    "CompositeLoss",
]

_EPS = 1e-7

def _to_probs(pred: torch.Tensor, from_logits: bool) -> torch.Tensor:
    return pred.sigmoid() if from_logits else pred.clamp(0.0, 1.0)

def _flatten(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Accept shapes [B, 1, H, W] or [B, H, W]
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred[:, 0]
    if target.dim() == 4 and target.size(1) == 1:
        target = target[:, 0]
    return pred.contiguous().view(pred.size(0), -1), target.contiguous().view(target.size(0), -1)

def wbce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: Optional[float] = None,
    from_logits: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Weighted BCE (supports logits). `pos_weight` > 1 increases positive class weight.
    pred: [B, 1, H, W] or [B, H, W]
    target: same shape, with {0,1}
    """
    if from_logits:
        if pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(pred, target.float(), pos_weight=torch.tensor(pos_weight, device=pred.device), reduction=reduction)
        else:
            loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction=reduction)
    else:
        # numerically stable BCE on probabilities
        p = pred.clamp(_EPS, 1.0 - _EPS)
        loss = F.binary_cross_entropy(p, target.float(), reduction=reduction)
        if pos_weight is not None:
            # Scale positive pixels
            w = torch.ones_like(target, device=pred.device)
            w = w + (pos_weight - 1.0) * target
            if reduction == "none":
                loss = loss * w
            else:
                loss = (loss * w).sum() / (w.sum() + _EPS)
    return loss

def soft_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    from_logits: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    1 - Soft Dice over batch mean.
    """
    p = _to_probs(pred, from_logits)
    p, t = _flatten(p, target.float())
    intersection = (p * t).sum(dim=1)
    denom = p.sum(dim=1) + t.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denom + smooth + _EPS)
    loss = 1.0 - dice
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

def iou_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    from_logits: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    1 - Jaccard (IoU) on probabilities.
    """
    p = _to_probs(pred, from_logits)
    p, t = _flatten(p, target.float())
    intersection = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1) - intersection
    jacc = (intersection + smooth) / (union + smooth + _EPS)
    loss = 1.0 - jacc
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

def _gaussian_kernel1d(kernel_size: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device).float() - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / (g.sum() + _EPS)
    return g

def _gaussian_filter(window_size: int, sigma: float, channels: int, device) -> torch.Tensor:
    g1d = _gaussian_kernel1d(window_size, sigma, device)
    g2d = torch.outer(g1d, g1d)
    kernel = g2d.expand(channels, 1, window_size, window_size).contiguous()
    return kernel

def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
    sigma: float = 1.5,
    from_logits: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Single-scale SSIM loss = 1 - SSIM on probabilities.
    Operates per-sample, then reduces.
    """
    p = _to_probs(pred, from_logits)
    t = target.float().clamp(0.0, 1.0)
    B = p.shape[0]
    # Ensure shape [B,1,H,W]
    if p.dim() == 3:
        p = p.unsqueeze(1)
    if t.dim() == 3:
        t = t.unsqueeze(1)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    padding = window_size // 2
    kernel = _gaussian_filter(window_size, sigma, channels=1, device=p.device)
    mu_p = F.conv2d(p, kernel, padding=padding, groups=1)
    mu_t = F.conv2d(t, kernel, padding=padding, groups=1)
    sigma_p = F.conv2d(p * p, kernel, padding=padding, groups=1) - mu_p * mu_p
    sigma_t = F.conv2d(t * t, kernel, padding=padding, groups=1) - mu_t * mu_t
    sigma_pt = F.conv2d(p * t, kernel, padding=padding, groups=1) - mu_p * mu_t

    ssim_map = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / ((mu_p**2 + mu_t**2 + C1) * (sigma_p + sigma_t + C2) + _EPS)
    # Reduce per-sample
    loss_map = 1.0 - ssim_map.clamp(0.0, 1.0)
    loss = loss_map.mean(dim=(1, 2, 3))  # [B]
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

def boundary_bce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    edge_weight: float = 2.0,
    from_logits: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Boundary-weighted BCE. Emphasizes pixels lying on GT edges (Sobel).
    edge_weight: additional weight applied to boundary pixels (>=1).
    """
    # Build Sobel kernels
    device = pred.device
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    t = target.float()
    if t.dim() == 3:
        t = t.unsqueeze(1)

    gx = F.conv2d(t, sobel_x, padding=1)
    gy = F.conv2d(t, sobel_y, padding=1)
    grad_mag = (gx.abs() + gy.abs())  # L1 magnitude
    boundary = (grad_mag > 0).float()  # binary edge map

    # Weight map: 1 for non-edge, (1 + edge_weight) for edge pixels
    weight = 1.0 + edge_weight * boundary
    if from_logits:
        bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
    else:
        p = pred.clamp(_EPS, 1.0 - _EPS)
        bce = F.binary_cross_entropy(p, target.float(), reduction="none")

    if bce.dim() == 4 and bce.size(1) == 1:
        bce = bce[:, 0]
        weight = weight[:, 0]
    elif bce.dim() == 3:
        # already [B,H,W]
        weight = weight[:, 0]

    # Normalize weighted average to avoid scale drift
    loss = (bce * weight).sum(dim=(1, 2)) / (weight.sum(dim=(1, 2)) + _EPS)  # [B]
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
def soft_boundary_bce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    edge_weight: float = 2.0,
    gamma: float = 1.0,
    from_logits: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Soft boundary-weighted BCE. Compared to the hard version：
      - 不再用 (grad>0) 变成 0/1 边界，而是用连续的 grad_mag 作为 soft 权重；
      - 可用 `gamma>1` 让边界更细一点。
    edge_weight: 叠加到边界上的最大额外权重（当 soft_boundary=1 时）。
    gamma: 用于 soft_boundary = (norm_grad)**gamma，gamma>1 会让边更细。
    """
    device = pred.device

    # Sobel kernels
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    t = target.float()
    if t.dim() == 3:
        t = t.unsqueeze(1)  # [B,1,H,W]

    # 计算连续的边缘强度
    gx = F.conv2d(t, sobel_x, padding=1)
    gy = F.conv2d(t, sobel_y, padding=1)
    grad_mag = gx.abs() + gy.abs()  # [B,1,H,W]

    # per-sample 归一化到 [0,1]，避免某些图 grad 很大/很小
    B = grad_mag.size(0)
    grad_flat = grad_mag.view(B, -1)                          # [B, H*W]
    max_per_img = grad_flat.max(dim=1, keepdim=True).values   # [B,1]
    max_per_img = max_per_img.clamp(min=1e-6)                 # 防止除 0
    norm_grad = (grad_flat / max_per_img).view_as(grad_mag)   # [B,1,H,W] in [0,1]

    # (可选) 幂次变换，让边界更“细”
    if gamma != 1.0:
        norm_grad = norm_grad.pow(gamma)

    soft_boundary = norm_grad  # [B,1,H,W], 0~1

    # 最终权重：非边界附近 ≈1，真正边界 ≈ 1 + edge_weight
    weight = 1.0 + edge_weight * soft_boundary

    # BCE 部分
    if from_logits:
        bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
    else:
        p = pred.clamp(1e-7, 1.0 - 1e-7)
        bce = F.binary_cross_entropy(p, target.float(), reduction="none")

    if bce.dim() == 4 and bce.size(1) == 1:
        bce = bce[:, 0]
        weight = weight[:, 0]
    elif bce.dim() == 3:
        weight = weight[:, 0]

    # 归一化的加权平均
    loss = (bce * weight).sum(dim=(1, 2)) / (weight.sum(dim=(1, 2)) + 1e-7)  # [B]
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

def soft_fbeta_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta2: float = 0.3,
    from_logits: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    1 - differentiable F_beta (default beta^2=0.3 to match maxF in your eval).
    Computed per-sample across all pixels and averaged by `reduction`.
    """
    p = _to_probs(pred, from_logits)
    p, t = _flatten(p, target.float())

    # Per-sample stats
    tp = (p * t).sum(dim=1)                       # [B]
    prec = tp / (p.sum(dim=1) + _EPS)             # [B]
    rec  = tp / (t.sum(dim=1) + _EPS)             # [B]
    fbeta = (1.0 + beta2) * prec * rec / (beta2 * prec + rec + _EPS)
    loss = 1.0 - fbeta.clamp(0.0, 1.0)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

class CompositeLoss(nn.Module):
    """
    Default weights are tuned for VT5000/VT821/VT1000 saliency:
      L = 1.0*WBCE + 0.5*Dice + 0.5*IoU + 0.3*SSIM + 0.2*BoundaryBCE + 0.5*SoftF_beta
    You can set from_logits=False if your model already outputs probabilities.
    """
    def __init__(
        self,
        wbce_w: float = 0.0,
        dice_w: float = 0.0,
        iou_w: float = 0.0,
        ssim_w: float = 0.0,
        boundary_w: float = 0.0,
        soft_bce: float = 0.0,
        edge_weight: float = 1.0,
        gamma: float = 2.0,
        fbeta_w: float = 0.0,
        pos_weight: Optional[float] = 0.0,  # emphasize positives for small objects
        beta2: float = 0.0,
        from_logits: bool = True,
    ):
        super().__init__()
        self.wbce_w = wbce_w
        self.dice_w = dice_w
        self.iou_w = iou_w
        self.ssim_w = ssim_w
        self.boundary_w = boundary_w
        self.fbeta_w = fbeta_w
        self.pos_weight = pos_weight
        self.beta2 = beta2
        self.from_logits = from_logits
        self.soft_bce = soft_bce
        self.edge_weight = edge_weight
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = {}

        if self.wbce_w:
            losses["wbce"] = wbce_loss(pred, target, pos_weight=self.pos_weight, from_logits=self.from_logits, reduction="mean")
        if self.dice_w:
            losses["dice"] = soft_dice_loss(pred, target, from_logits=self.from_logits, reduction="mean")
        if self.iou_w:
            losses["iou"] = iou_loss(pred, target, from_logits=self.from_logits, reduction="mean")
        if self.ssim_w:
            losses["ssim"] = ssim_loss(pred, target, from_logits=self.from_logits, reduction="mean")
        if self.boundary_w:
            losses["boundary"] = boundary_bce_loss(pred, target, from_logits=self.from_logits, reduction="mean")
        if self.fbeta_w:
            losses["fbeta"] = soft_fbeta_loss(pred, target, beta2=self.beta2, from_logits=self.from_logits, reduction="mean")
        if self.soft_bce:
            losses["soft_bce"] = soft_boundary_bce_loss(pred, target, edge_weight=self.edge_weight,
                                                      gamma=self.gamma, from_logits=self.from_logits, reduction="mean")


        total = sum([
            self.wbce_w * losses.get("wbce", 0.0),
            self.dice_w * losses.get("dice", 0.0),
            self.iou_w * losses.get("iou", 0.0),
            self.ssim_w * losses.get("ssim", 0.0),
            self.boundary_w * losses.get("boundary", 0.0),
            self.fbeta_w * losses.get("fbeta", 0.0),
            self.soft_bce * losses.get("soft_bce", 0.0),
        ])

        # For logging
        self.last_components = {k: v.detach() if torch.is_tensor(v) else v for k, v in losses.items()}
        return total
        
class LegacyIoUDiceLoss(nn.Module):
    """
    完全复刻你原来的:
      loss = iou_loss(pred, gt) + dice_loss(pred, gt)
    只是方便之后再叠加一点点别的项。
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # IoU 部分
        p = torch.sigmoid(pred)
        m = target.float()
        inter = (p * m).sum(dim=(2, 3))
        union = (p + m).sum(dim=(2, 3))
        iou = 1.0 - (inter + 1.0) / (union - inter + 1.0)
        iou_loss_val = iou.mean()

        # Dice 部分（整 batch flatten）
        p_flat = p.view(-1)
        m_flat = m.view(-1)
        intersection = (p_flat * m_flat).sum()
        dice = (2. * intersection + 1.0) / (p_flat.sum() + m_flat.sum() + 1.0)
        dice_loss_val = 1.0 - dice

        return iou_loss_val + dice_loss_val

class LegacyIoUDiceLossbce(nn.Module):
    """
    完全复刻你原来的:
      loss = iou_loss(pred, gt) + dice_loss(pred, gt)
    只是方便之后再叠加一点点别的项。
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # IoU 部分
        p = torch.sigmoid(pred)
        m = target.float()
        inter = (p * m).sum(dim=(2, 3))
        union = (p + m).sum(dim=(2, 3))
        iou = 1.0 - (inter + 1.0) / (union - inter + 1.0)
        iou_loss_val = iou.mean()

        # Dice 部分（整 batch flatten）
        p_flat = p.view(-1)
        m_flat = m.view(-1)
        intersection = (p_flat * m_flat).sum()
        dice = (2. * intersection + 1.0) / (p_flat.sum() + m_flat.sum() + 1.0)
        dice_loss_val = 1.0 - dice

        return iou_loss_val + dice_loss_val + 0.1 * boundary_bce_loss(pred, target, edge_weight=2.0, from_logits=True, reduction="mean")
    
class LegacyIoUDiceLosssoftbce(nn.Module):
    """
    完全复刻你原来的:
      loss = iou_loss(pred, gt) + dice_loss(pred, gt)
    只是方便之后再叠加一点点别的项。
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # IoU 部分
        p = torch.sigmoid(pred)
        m = target.float()
        inter = (p * m).sum(dim=(2, 3))
        union = (p + m).sum(dim=(2, 3))
        iou = 1.0 - (inter + 1.0) / (union - inter + 1.0)
        iou_loss_val = iou.mean()

        # Dice 部分（整 batch flatten）
        p_flat = p.view(-1)
        m_flat = m.view(-1)
        intersection = (p_flat * m_flat).sum()
        dice = (2. * intersection + 1.0) / (p_flat.sum() + m_flat.sum() + 1.0)
        dice_loss_val = 1.0 - dice

        return iou_loss_val + dice_loss_val + 0.1 * soft_boundary_bce_loss(pred, target, edge_weight=2.0, gamma=1.5,
                         from_logits=True, reduction="mean")