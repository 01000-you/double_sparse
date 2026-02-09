"""
2SSP + Double Sparse Factorization 결합 알고리즘

프로세스 (사용자 의도):
1. W를 AB로 분해 (double_sparse 초기 분해)
2. 2SSP 알고리즘으로 AB 사이 어떤 채널을 reduction할지 결정
3. 채널 마스크 고정 후 ADMM으로 최적화
"""
import math
import time
import torch
import torch.nn as nn
import numpy as np
from doublesparse import find_other2, mag_prune, finalize, factorizef, factorizeT

DEBUG = False
VERBOSE = False  # True면 err_prefin, err_fin, time 등 상세 출력
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def initial_factorize(W, XX, iters=15, verbose=False, name=""):
    """
    Step 1: W를 AB로 초기 분해 (채널 제약 없음). sparsity 미사용, dense 분해.
    Returns: W_approx (= A @ B)
    """
    asp = 0.16 if W.shape[0] == W.shape[1] else 0.25
    sp = 1.0  # dense
    if W.shape[0] >= W.shape[1]:
        W2, _, _ = factorizeT(W.T, XX, asp, sp=sp, iters=iters, fixmask=None, verbose=verbose, name=name)
        return W2.T
    W2, _, _ = factorizef(W, XX, asp=asp, sp=sp, iters=iters, fixmask=None, verbose=verbose, name=name)
    return W2


def _make_channel_fixmasks(channel_keep_mask, out_dim, in_dim, device):
    """
    채널 유지 마스크로부터 A, B용 fixmask 생성.
    W = A @ B, A [out, out], B [out, in] (factorizef)
    또는 W.T 분해 시 A [out, out], B [out, in] (factorizeT)
    채널 j 프루닝 -> A[:,j]=0, B[j,:]=0
    """
    mask = channel_keep_mask.float().to(device)
    fixmask_A = mask.unsqueeze(0).expand(out_dim, out_dim)
    fixmask_B = mask.unsqueeze(1).expand(out_dim, in_dim)
    return fixmask_A, fixmask_B


def factorize_with_channel_mask(W, XX, channel_keep_mask, iters=40, nofinal=False, channel_dim="out"):
    """
    W = A @ B 분해 + 채널 마스크 적용 ADMM.
    채널만 조절 - sparsity(요소 희소도) 무시, 유지 채널 내부는 dense.
    channel_keep_mask: [channel_dim] bool tensor, True=유지할 채널
    """
    asp = 0.16 if W.shape[0] == W.shape[1] else 0.25
    
    if channel_dim == "in":
        return _factorize_in_channel(W, XX, channel_keep_mask, asp, iters, nofinal)
    elif W.shape[0] >= W.shape[1]:
        return _factorizeT_channel(W, XX, channel_keep_mask, asp, iters, nofinal)
    else:
        return _factorizef_channel(W, XX, channel_keep_mask, asp, iters, nofinal)


def _factorizef_channel(W, XX, channel_keep_mask, asp, iters, nofinal):
    """W [out, in], out < in. A [out, out], B [out, in]. 채널 = out. 채널만 pruning (sparsity 무시)."""
    out, in_dim = W.shape[0], W.shape[1]
    device = W.device
    
    fixmask_A, fixmask_B = _make_channel_fixmasks(channel_keep_mask, out, in_dim, device)
    nza = int((fixmask_A != 0).sum().item())
    nzb = int((fixmask_B != 0).sum().item())
    
    norm = XX.diag().sqrt() + 1e-8
    Wn = W * norm
    
    Az = torch.eye(out, device=device) * fixmask_A
    Au = torch.zeros_like(Az)
    Bz = Wn * fixmask_B
    Bu = torch.zeros_like(Bz)
    
    for itt in range(iters):
        rho_start = min(1.0, itt / max(1, iters - 3)) ** 3
        Az, Au = (x.T for x in find_other2(Bz.T, Wn.T, nza, Az.T, Au.T, reg=1e-2, 
            debug=False, rho_start=rho_start, fixmask=fixmask_A.T))
        Az = Az * fixmask_A
        Bz, Bu = find_other2(Az, Wn, nzb, Bz, Bu, reg=1e-2, 
            debug=False, rho_start=rho_start, fixmask=fixmask_B)
        Bz = Bz * fixmask_B
    
    W2 = Az.matmul(Bz / norm)
    if VERBOSE:
        print("err_prefin", (W2 - W).matmul(XX).matmul((W2 - W).T).diag().sum().item())
    if nofinal:
        return W2, Az.cpu(), (Bz / norm).cpu()
    Ac = finalize(XX, W, Az, Bz / norm)
    W3 = Ac.matmul(Bz / norm)
    if VERBOSE:
        print("err_fin   ", (W3 - W).matmul(XX).matmul((W3 - W).T).diag().sum().item())
    return W3, Ac.cpu(), (Bz / norm).cpu()


def _factorize_in_channel(W, XX, channel_keep_mask, asp, iters, nofinal):
    """W [out, in], 채널 = in. down_proj용. 채널만 pruning (sparsity 무시)."""
    out, in_dim = W.shape[0], W.shape[1]
    device = W.device

    fixmask_A = channel_keep_mask.unsqueeze(0).expand(out, in_dim).float().to(device)
    fixmask_B = channel_keep_mask.unsqueeze(1).expand(in_dim, in_dim).float().to(device)

    nza = int((fixmask_A != 0).sum().item())
    nzb = int((fixmask_B != 0).sum().item())

    norm = XX.diag().sqrt() + 1e-8
    Wn = W * norm

    Az = Wn * fixmask_A
    Au = torch.zeros_like(Az)
    Bz = torch.eye(in_dim, device=device) * fixmask_B
    Bu = torch.zeros_like(Bz)

    for itt in range(iters):
        rho_start = min(1.0, itt / max(1, iters - 3)) ** 3
        Az, Au = (x.T for x in find_other2(Bz.T, Wn.T, nza, Az.T, Au.T, reg=1e-2,
            debug=False, rho_start=rho_start, fixmask=fixmask_A.T))
        Az = Az * fixmask_A
        Bz, Bu = find_other2(Az, Wn, nzb, Bz, Bu, reg=1e-2,
            debug=False, rho_start=rho_start, fixmask=fixmask_B)
        Bz = Bz * fixmask_B

    W2 = Az.matmul(Bz / norm)
    if VERBOSE:
        print("err_prefin", (W2 - W).matmul(XX).matmul((W2 - W).T).diag().sum().item())
    if nofinal:
        return W2, Az.cpu(), (Bz / norm).cpu()
    Ac = finalize(XX, W, Az, Bz / norm)
    W3 = Ac.matmul(Bz / norm)
    if VERBOSE:
        print("err_fin   ", (W3 - W).matmul(XX).matmul((W3 - W).T).diag().sum().item())
    return W3, Ac.cpu(), (Bz / norm).cpu()


def _factorizeT_channel(W, XX, channel_keep_mask, asp, iters, nofinal):
    """W [out, in], out >= in. 채널만 pruning (sparsity 무시)."""
    out, in_dim = W.shape[0], W.shape[1]
    device = W.device
    W_T = W.T
    
    fixmask_A, fixmask_B = _make_channel_fixmasks(channel_keep_mask, out, in_dim, device)
    nza = int((fixmask_A != 0).sum().item())
    nzb = int((fixmask_B != 0).sum().item())
    
    norm = XX.diag().sqrt().unsqueeze(1) + 1e-8
    Wn = W_T * norm
    
    Az = torch.eye(out, device=device) * fixmask_A
    Au = torch.zeros_like(Az)
    Bz = Wn * fixmask_B
    Bu = torch.zeros_like(Bz)
    
    for itt in range(iters):
        rho_start = min(1.0, itt / max(1, iters - 3)) ** 3
        Az, Au = (x.T for x in find_other2(Bz.T, Wn.T, nza, Az.T, Au.T, reg=1e-2,
            debug=False, rho_start=rho_start, fixmask=fixmask_A.T))
        Az = Az * fixmask_A
        Bz, Bu = find_other2(Az, Wn, nzb, Bz, Bu, reg=1e-2,
            debug=False, rho_start=rho_start, fixmask=fixmask_B)
        Bz = Bz * fixmask_B
    
    W2 = (Az / norm).matmul(Bz)
    if VERBOSE:
        print("err_prefin", (W2 - W).matmul(XX).matmul((W2 - W).T).diag().sum().item())
    if nofinal:
        return W2, (Az / norm).cpu(), Bz.cpu()
    Ac = finalize(XX, W, (Az / norm).T, Bz)
    W3 = Ac.matmul(Bz)
    if VERBOSE:
        print("err_fin   ", (W3 - W).matmul(XX).matmul((W3 - W).T).diag().sum().item())
    return W3, Ac.cpu(), Bz.cpu()


class DoubleSparse2SSP:
    """
    2SSP 채널 선택 + Double Sparse ADMM 결합 클래스.
    channel_dim: "out" (up_proj, gate_proj) or "in" (down_proj)
    """
    def __init__(self, layer, channel_keep_mask=None, nofinal=False, channel_dim="out"):
        self.layer = layer
        self.dev = layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1)
        self.rows, self.columns = W.shape[0], W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.nofinal = nofinal
        self.channel_keep_mask = channel_keep_mask
        self.channel_dim = channel_dim

    def add_batch(self, inp, out, blocksize=1024):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(self, channel_keep_mask=None):
        if channel_keep_mask is not None:
            self.channel_keep_mask = channel_keep_mask
        if self.channel_keep_mask is None:
            raise ValueError("channel_keep_mask must be provided")
        
        W = self.layer.weight.data.clone().float()
        tick = time.time()
        W2, _, _ = factorize_with_channel_mask(
            W, self.H, self.channel_keep_mask,
            nofinal=self.nofinal, channel_dim=self.channel_dim
        )
        torch.cuda.synchronize()
        if VERBOSE:
            print('time %.2f' % (time.time() - tick))
        self.layer.weight.data = W2.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


def compute_channel_importance_2ssp(model, calibration_batches, layer_indices, intermediate_size):
    """
    2SSP 스타일: MLP hidden state (down_proj 입력) L2 norm으로 채널 중요도 계산.
    Returns: dict layer_idx -> [intermediate_size] tensor (채널별 L2 norm)
    """
    from tqdm import tqdm
    device = next(model.parameters()).device
    
    for i, layer in enumerate(model.model.layers):
        layer.mlp.down_proj._2ssp_layer_idx = i
    
    hidden_states_per_layer = {i: [] for i in layer_indices}
    
    def hook(module, input, output):
        if module._2ssp_layer_idx in layer_indices:
            hidden_states_per_layer[module._2ssp_layer_idx].append(input[0].detach())
    
    hooks = []
    for i in layer_indices:
        hooks.append(model.model.layers[i].mlp.down_proj.register_forward_hook(hook))
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(calibration_batches, desc="Channel importance"):
            inp = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            if inp.dim() == 3:
                inp = inp.view(-1, inp.size(-1))
            _ = model(inp)
    
    for h in hooks:
        h.remove()
    
    channel_norms = {}
    for li in layer_indices:
        if hidden_states_per_layer[li]:
            stacked = torch.cat([h for h in hidden_states_per_layer[li]], dim=0)
            norms = stacked.norm(dim=0, p=2)
            channel_norms[li] = norms
        else:
            channel_norms[li] = torch.ones(intermediate_size, device=device)
    
    return channel_norms


def compute_num_keep_from_target_compression(target_compression, hidden_size, intermediate_size):
    """
    W 대비 (A+B) 저장 시 목표 압축률에 맞는 num_keep 계산.
    target_compression=0.3: 30% 압축 → 저장 = 0.7*W
    (A+B saved) = (1 - target_compression) * W
    """
    keep_ratio = 1.0 - target_compression
    # MLP: up, gate, down 모두 channel=intermediate. W=3*h*im, A+B=3*(im²+im*h)
    num_keep_mlp = max(1, int(
        intermediate_size * keep_ratio * hidden_size / (intermediate_size + hidden_size)
    ))
    # Attn: W=4*h², A+B=8*h² per layer
    num_keep_attn = max(1, int(hidden_size * keep_ratio * 0.5))
    return num_keep_mlp, num_keep_attn


def get_channel_keep_mask(channel_norms, num_keep):
    """상위 num_keep개 채널 유지 마스크"""
    _, top_indices = torch.topk(channel_norms, min(num_keep, channel_norms.numel()))
    mask = torch.zeros_like(channel_norms, dtype=torch.bool, device=channel_norms.device)
    mask[top_indices] = True
    return mask
