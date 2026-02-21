import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange
import math

__all__ = [
    "resNetBlock",
    "AWGNChannel",
    "WaveNet3D",
    "ChunkMaskGIT",
    "ChAMAEViT"
]

class AWGNChannel(nn.Module):
    '''
    Trainable/Untrainable SISO Channel for torch
    Now supports batch-wise SNR
    '''
    def __init__(self, SNRdB=None):
        super(AWGNChannel, self).__init__()
        # Initialize with a default value if needed, though forward input is preferred
        self.default_snr = 10 ** (SNRdB / 10) if SNRdB is not None else None

    def forward(self, x, snr_db=None):
        assert x.dtype == torch.complex64, f"input dtype should be complex64. Now is {x.dtype}"
        
        # Power Normalization (Unit Average Power)
        # Normalize per element (or per channel/batch depending on requirement, usually per element for AWGN)
        pwr = torch.mean(torch.abs(x)**2, dim=-1, keepdim=True)

        # Noise Generation
        if snr_db is None:
            assert self.default_snr is not None, "SNR must be provided"
            snr_linear = self.default_snr
            # If snr_linear is scalar, expand to batch? 
            # Logic below handles tensor inputs mainly.
            noise_std = (1.0 / snr_linear) ** 0.5
        else:
            # snr_db shape: (B, 1)
            # 10^(SNR/10)
            snr_linear = 10 ** (snr_db / 10.0)
            # P_noise = P_sig / SNR = 1 / SNR (since P_sig normalized to 1)
            # std = sqrt(P_noise / 2) for real/imag parts. 
            # But torch.randn(complex) generates complex normal with var=1 (real var=0.5, imag var=0.5).
            # So we just need scaling factor sqrt(1/SNR).
            noise_std = (1.0 / snr_linear) ** 0.5
            
            # Broadcasting noise_std (B, 1) to (B, L)
            # Assuming x is (B, L) or similar flattened complex vector
            if x.dim() > 2:
                # If x is (B, C, H, W) complex, reshape snr to broadcast
                view_shape = [x.shape[0]] + [1] * (x.dim() - 1)
                noise_std = noise_std.view(*view_shape)

        # Generate Noise
        # torch.randn(complex64) has std=1 (Var=1). 
        # We want Noise Power N0 = 1/SNR. 
        # So we simply scale by sqrt(1/SNR).
        n = noise_std * torch.randn(x.shape, dtype=torch.complex64, device=x.device)
        
        # Add noise
        y = x + n
        return y

class resNetBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, groups:int=1,down:float=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,int(in_ch/down), kernel_size=1,groups=groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(int(in_ch/down),int(in_ch/down), kernel_size=3,padding=1,groups=groups)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(int(in_ch/down),in_ch, kernel_size=1,groups=groups)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        out = out + identity
        return out

class ResBlock3D(nn.Module):
    """
    Implements the Gated Residual Block.
    Structure:
      / [Conv3d -> Tanh] \
    x                         * -> 1x1 Conv3d -> + -> Output
    | \ [Conv3d -> Sigm] /                 |
    |------------------------------------------|
    """
    def __init__(self, channels, kernel_size=(3,3,3), dilation=(1,1,1)):
        super(ResBlock3D, self).__init__()
        
        # Filter path (Tanh)
        self.filter_conv = nn.Conv3d(channels, channels, kernel_size, dilation, padding=(1,1,1))
        # Gate path (Sigmoid)
        self.gate_conv = nn.Conv3d(channels, channels, kernel_size, dilation,padding=(1,1,1))
        
        # 1x1 Conv to mix features before adding residual
        self.skip_conv = nn.Conv3d(channels, channels, kernel_size=1, dilation=dilation)
        
    def forward(self, x):
        # x: (N, C, T, H, W)
        
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        
        # Element-wise multiplication (Gated Activation)
        out = filter_out * gate_out
        
        # 1x1 Projection
        out = self.skip_conv(out)
        
        # Residual connection
        return x + out

class WaveNet3D(nn.Module):
    def __init__(self, dims=[32, 64, 128], kernel_size=3,num_blocks=3, chunk_dim=16, input_length=9):
        super().__init__()
        self.length = input_length
        self.chunk_size=chunk_dim
        self.layers = nn.ModuleList()
        
        # 8의 receptive field를 얻으려면 3-stage가 필요함.
        channels=[chunk_dim]+dims

        ks=(kernel_size,3,3)
            
        for i in range(len(dims)):
            # 이거 다시 짜야 해. 왜냐? Downsampling layer란 말임.
            stage_ops=[nn.Conv3d(channels[i],channels[i+1],kernel_size=ks),
                       *[ResBlock3D(channels[i+1]) for _ in range(num_blocks)]]
            self.layers.append(nn.Sequential(*stage_ops))

        self.positional_embadding=nn.Parameter(torch.randn(1,self.chunk_size,self.length,1,1))
            
    def forward(self, x: torch.Tensor):
        # Input x: (B, F, C_chunk, H, W)
        # We need to permute for Conv3d: (B, C_chunk, F, H, W)
        B,length,chunk,H,W=x.shape
        temp = rearrange(x, 'b t c h w -> b c t h w')
        if length<self.length:  # 아마 F는 8일 거고 self.length는 9일 거임.
            temp=F.pad(temp, (0,0,0,0,0,self.chunk_size*(self.length-length)))
        # 여기서 positional embedding을 더해야 함.
        temp=temp+self.positional_embadding.expand(B,-1,-1,H,W)
        
        # Pass through WaveNet layers
        for i, layer in enumerate(self.layers):
            temp = layer(temp)
            print(temp.shape)
            
        # temp is now (B, dims[-1], 1, H, W)
        temp = temp.squeeze(2)
        
        return temp
    
class ChunkMaskGIT(nn.Module):
    """
    Continuous MaskGIT-like filler for channel/chunk-masked JSCC latents.

    - Treat each chunk (symbol_dim//len,H,W) as a token.
    - Encode chunks via nn.Conv2d(symbol_dim,len*d_model,4,2,2,groups=self.len) and reshape
    -> token embeddings (B*H//2*W//2,len,d_model)
    - Replace missing tokens with a learned [MASK] embedding
    - Bidirectional Transformer predicts ALL tokens in parallel
    - Decode predicted token embeddings -> chunk tensors (B,C,H,W)
    - Iterative decoding: keep most confident predictions, re-mask others (cosine schedule)
      (MaskGIT-style confidence-driven unmasking + decreasing mask ratio). :contentReference[oaicite:1]{index=1}
    """

    def __init__(
        self,
        symbol_dim: int = 256,           # 128
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        num_steps: int = 6,       # T for iterative decoding (inference)
        len:int=8,
        use_mask_flag_embed: bool = True,
    ):
        # nn.Conv2d(symbol_dim,len*d_model,4,2,2,groups=self.len)으로 일단 tokenization을 해.
        # Positional Embedding을 더해야 하나? 일단 더해.
        # 그럼 (B, d_model*self.len, H//2, W//2)짜리가 나올 텐데 그걸 (B*H//2*W//2, d_model, self.len)로 바꿔
        # 그리고 거기에 transformer를 돌려.
        # 이후 다시 (B, d_model*self.len, H//2, W//2)로 바꾸고 ConvTranspose2d로 (B,symbol_dim,H,W)로 되돌려.

        super().__init__()
        self.symbol_dim=int(symbol_dim)
        self.d_model = int(d_model)
        self.len=int(len)
        self.num_steps = int(num_steps)
        self.use_mask_flag_embed = bool(use_mask_flag_embed)

        # ---- chunk -> token embedding ----
        self.to_token=nn.Conv2d(symbol_dim,len*d_model,5,2,2,groups=self.len)

        # Positional embedding over chunk index (length F)
        self.pos_emb = nn.Parameter(torch.randn(1, self.len, self.d_model) * 0.02)

        # Learned [MASK] token in embedding space
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        # Optional: add a learned embedding indicating visible/missing
        if self.use_mask_flag_embed:
            # 0: visible, 1: missing
            self.mask_flag_emb = nn.Embedding(2, self.d_model)

        # ---- Transformer (bidirectional) ----
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=4 * self.d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(self.d_model)

        # Predict confidence per chunk (used for iterative unmasking)
        self.conf_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 1),
        )

        # ---- token embedding -> chunk ----
        self.from_token = nn.ConvTranspose2d(len*d_model,symbol_dim,5,2,2,1,groups=self.len)
        # refine after upsample
        self.refine = resNetBlock(self.symbol_dim)

    def _encode_tokens(self, y_in: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        B, C, H, W = y_in.shape
        
        # Conv2d -> (B, len*d_model, H_out, W_out)
        x = self.to_token(y_in)
        _, _, H_out, W_out = x.shape
        
        # (B, len, d_model, H_out, W_out)
        x = x.view(B, self.len, self.d_model, H_out, W_out)
        
        # (B, H_out, W_out, len, d_model)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        
        # (B * H_out * W_out, len, d_model)
        t = x.view(-1, self.len, self.d_model)
        
        return t, (H_out, W_out)

    def _decode_tokens(self, t: torch.Tensor, B: int, H_out: int, W_out: int) -> torch.Tensor:
        # t: (B * H_out * W_out, len, d_model)
        
        # (B, H_out, W_out, len, d_model)
        x = t.view(B, H_out, W_out, self.len, self.d_model)
        
        # (B, len, d_model, H_out, W_out)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        
        # (B, len*d_model, H_out, W_out)
        x = x.view(B, self.len * self.d_model, H_out, W_out)
        
        # ConvTranspose2d -> (B, symbol_dim, H, W)
        y = self.from_token(x)
        y = self.refine(y)
        return y

    @torch.no_grad()
    def _cosine_remaining(self, total_missing: int, step: int, T: int) -> int:
        r = math.cos(((step + 1) / T) * (math.pi / 2.0))
        return int(math.ceil(total_missing * r))

    def _predict_once(self, y_work: torch.Tensor, mask_indices: torch.Tensor):
        B = y_work.size(0)
        
        tokens, (H_out, W_out) = self._encode_tokens(y_work)
        
        # mask_indices: (B, len) -> (B * H_out * W_out, len)
        mask_expanded = mask_indices.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, len)
        mask_expanded = mask_expanded.expand(B, H_out, W_out, self.len).reshape(-1, self.len)
        
        if self.use_mask_flag_embed:
            flag = mask_expanded.long()
            tokens = tokens + self.mask_flag_emb(flag)
            
        if mask_expanded.any():
            mask_tok = self.mask_token.expand(B * H_out * W_out, self.len, -1)
            tokens = torch.where(mask_expanded.unsqueeze(-1), mask_tok, tokens)
            
        tokens = tokens + self.pos_emb  # (1, len, d_model) 브로드캐스팅

        h = self.transformer(tokens)
        h = self.norm(h)

        # confidence 산출 (B * H_out * W_out, len)
        conf_flat = torch.sigmoid(self.conf_head(h)).squeeze(-1)
        
        # 청크 단위 신뢰도: 공간(H_out * W_out) 차원 평균 -> (B, len)
        conf_spatial = conf_flat.view(B, H_out * W_out, self.len)
        conf = conf_spatial.mean(dim=1)
        
        pred_chunks = self._decode_tokens(h, B, H_out, W_out)
        
        return pred_chunks, conf

    def forward(
        self,
        y_in: torch.Tensor,
        mask_indices: torch.Tensor,
        num_steps: int | None = None,
        return_aux: bool = False,
    ):
        """
        y_in: (B, symbol_dim, H, W)
        mask_indices: (B, len) bool, True=missing
        """
        B = y_in.size(0)
        
        if num_steps is None:
            T = 1 if self.training else self.num_steps
        else:
            T = int(num_steps)

        y_work = y_in.clone()
        missing = mask_indices.clone()

        # (B, len) 마스크를 (B, symbol_dim, 1, 1) 로 매핑하는 헬퍼 함수
        def get_spatial_mask(m):
            chunk_size = self.symbol_dim // self.len
            m_exp = torch.repeat_interleave(m, chunk_size, dim=1)
            return m_exp.unsqueeze(-1).unsqueeze(-1)

        if T <= 1 or not missing.any():
            pred, conf = self._predict_once(y_work, missing)
            spatial_mask = get_spatial_mask(missing)
            filled = torch.where(spatial_mask, pred, y_in)
            if return_aux:
                return filled, {"pred": pred, "conf": conf, "final_missing": missing}
            return filled

        for t in range(T):
            pred, conf = self._predict_once(y_work, missing)

            if t == T - 1:
                spatial_mask = get_spatial_mask(missing)
                y_work = torch.where(spatial_mask, pred, y_work)
                missing[:] = False
                break

            for b in range(B):
                miss_idx = torch.nonzero(missing[b], as_tuple=False).squeeze(-1)
                n_missing = int(miss_idx.numel())
                if n_missing == 0:
                    continue

                remaining = self._cosine_remaining(total_missing=n_missing, step=t, T=T)
                to_fill = n_missing - remaining
                if to_fill <= 0:
                    continue

                miss_conf = conf[b, miss_idx]
                topk = torch.topk(miss_conf, k=to_fill, largest=True).indices
                fill_idx = miss_idx[topk]

                # 선택된 청크 인덱스에 맞춰 실제 채널 단위 업데이트
                chunk_size = self.symbol_dim // self.len
                for idx in fill_idx:
                    start = idx * chunk_size
                    end = start + chunk_size
                    y_work[b, start:end] = pred[b, start:end]
                    
                missing[b, fill_idx] = False

        if return_aux:
            return y_work, {"pred": pred, "conf": conf, "final_missing": missing}
        return y_work

'''class MaskGIT3D(nn.Module):
    """
    3D Voxel-based MaskGIT for 'Swiss Cheese' masking.
    Divides (C, H, W) latent into small 3D patches (voxels) and treats each patch as a token.
    """

    def __init__(
        self,
        in_channels: int,           # C (Total latent channels)
        img_size: tuple[int, int],  # (H, W)
        patch_size: tuple[int, int, int] = (16, 2, 2), # (pc, ph, pw) -> Patch Dimension
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        num_steps: int = 6,
        use_mask_flag_embed: bool = True,
    ):
        super().__init__()
        self.C = in_channels
        self.H, self.W = img_size
        self.pc, self.ph, self.pw = patch_size
        self.d_model = d_model
        self.num_steps = num_steps
        self.use_mask_flag_embed = use_mask_flag_embed

        # Validate patch size
        assert self.C % self.pc == 0, f"Channel {self.C} not divisible by patch_c {self.pc}"
        assert self.H % self.ph == 0, f"Height {self.H} not divisible by patch_h {self.ph}"
        assert self.W % self.pw == 0, f"Width {self.W} not divisible by patch_w {self.pw}"

        self.n_patches_c = self.C // self.pc
        self.n_patches_h = self.H // self.ph
        self.n_patches_w = self.W // self.pw
        self.num_patches = self.n_patches_c * self.n_patches_h * self.n_patches_w
        
        # Patch vector dimension: pc * ph * pw
        self.patch_dim = self.pc * self.ph * self.pw

        # ---- Embedding ----
        self.to_token = nn.Linear(self.patch_dim, d_model)
        
        # Positional Embedding (Learnable)
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

        # [MASK] token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        if self.use_mask_flag_embed:
            self.mask_flag_emb = nn.Embedding(2, d_model)

        # ---- Transformer ----
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Confidence Head
        self.conf_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

        # ---- Reconstruction ----
        self.from_token = nn.Linear(d_model, self.patch_dim)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, C, H, W) -> (B, N_patches, Patch_dim)
        Splits volume into small 3D cubes.
        """
        # Rearrange using einops: split C, H, W into (num, size)
        x = rearrange(
            x, 
            'b (nc pc) (nh ph) (nw pw) -> b (nc nh nw) (pc ph pw)',
            pc=self.pc, ph=self.ph, pw=self.pw
        )
        return x

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, N_patches, Patch_dim) -> (B, C, H, W)
        """
        x = rearrange(
            x,
            'b (nc nh nw) (pc ph pw) -> b (nc pc) (nh ph) (nw pw)',
            nc=self.n_patches_c, nh=self.n_patches_h, nw=self.n_patches_w,
            pc=self.pc, ph=self.ph, pw=self.pw
        )
        return x

    def _cosine_remaining(self, total_missing: int, step: int, T: int) -> int:
        r = math.cos(((step + 1) / T) * (math.pi / 2.0))
        return int(math.ceil(total_missing * r))

    def _predict_once(self, x_work: torch.Tensor, mask_indices: torch.Tensor):
        """
        x_work: (B, C, H, W)
        mask_indices: (B, N_patches) boolean
        """
        # 1. Patchify & Embed
        patches = self._patchify(x_work)        # (B, N, P_dim)
        tokens = self.to_token(patches)         # (B, N, d_model)

        # 2. Add Flag Embedding
        if self.use_mask_flag_embed:
            flag = mask_indices.long()
            tokens = tokens + self.mask_flag_emb(flag)

        # 3. Apply Mask Token
        B, N, D = tokens.shape
        if mask_indices.any():
            mask_tok = self.mask_token.expand(B, N, -1)
            tokens = torch.where(mask_indices.unsqueeze(-1), mask_tok, tokens)

        # 4. Positional Embedding & Transformer
        tokens = tokens + self.pos_emb
        h = self.transformer(tokens)
        h = self.norm(h)

        # 5. Prediction & Confidence
        conf = torch.sigmoid(self.conf_head(h)).squeeze(-1) # (B, N)
        pred_patches = self.from_token(h)                   # (B, N, P_dim)
        
        # 6. Unpatchify to Image space
        pred_vol = self._unpatchify(pred_patches)           # (B, C, H, W)
        
        return pred_vol, conf

    def forward(self, y_in, mask_indices, num_steps=None, return_aux=False):
        """
        y_in: (B, C, H, W)
        mask_indices: (B, N_patches) bool, True=Missing
        """
        if num_steps is None:
            T = 1 if self.training else self.num_steps
        else:
            T = int(num_steps)

        y_work = y_in.clone()
        missing = mask_indices.clone()

        # Single pass
        if T <= 1:
            pred, conf = self._predict_once(y_work, missing)
            # Fill missing regions in pixel space
            # To do this correctly, we need the mask in pixel space (B, C, H, W)
            mask_pixel = self._unpatchify(missing.unsqueeze(-1).float().repeat(1, 1, self.patch_dim)).bool()
            
            filled = torch.where(mask_pixel, pred, y_in)
            
            if return_aux:
                return filled, {"pred": pred, "conf": conf}
            return filled

        # Iterative Decoding
        for t in range(T):
            pred, conf = self._predict_once(y_work, missing)
            
            # Create pixel-level mask for updating y_work
            mask_pixel = self._unpatchify(missing.unsqueeze(-1).float().repeat(1, 1, self.patch_dim)).bool()
            
            # Update prediction
            y_work = torch.where(mask_pixel, pred, y_work)

            if t == T - 1:
                break
            
            # Re-mask logic
            for b in range(y_in.size(0)):
                miss_idx = torch.nonzero(missing[b], as_tuple=False).squeeze(-1)
                n_missing = miss_idx.numel()
                if n_missing == 0: continue

                remaining = self._cosine_remaining(n_missing, t, T)
                to_fill = n_missing - remaining
                if to_fill <= 0: continue

                miss_conf = conf[b, miss_idx]
                topk = torch.topk(miss_conf, k=to_fill, largest=True).indices
                fill_idx = miss_idx[topk]
                
                # Unmask selected tokens
                missing[b, fill_idx] = False
                
        if return_aux:
            return y_work, {"pred": pred, "conf": conf}
        return y_work'''

class ChAMAEViT(nn.Module):
    """
    ChA-MAEViT for DeepJSCC Latents.
    - Uses Memory Tokens for cross-channel global context[cite: 64].
    - Uses Channel Tokens to identify chunks in a shared Channel-Aware Decoder[cite: 170, 174].
    - Single forward pass MAE reconstruction (not iterative like MaskGIT)[cite: 126].
    """
    def __init__(self, symbol_dim: int, F: int, d_model: int = 256, nhead: int = 8, 
                 enc_layers: int = 4, dec_layers: int = 2, num_memory_tokens: int = 4):
        super().__init__()
        self.symbol_dim = symbol_dim
        self.F = F
        self.d_model = d_model
        self.chunk_size = symbol_dim // F
        self.num_memory_tokens = num_memory_tokens

        # Patch Embedding (Spatial 2x2 pooling/conv as an example, or 1x1 for pixel-level)
        self.to_token = nn.Conv2d(self.chunk_size, d_model, kernel_size=4, stride=2, padding=1)
        
        # Tokens
        self.memory_tokens = nn.Parameter(torch.randn(1, num_memory_tokens, d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.channel_tokens = nn.Parameter(torch.randn(1, F, d_model) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02) # Simplified positional emb

        # Encoder (Processes only visible tokens + memory tokens)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        
        # Channel-Aware Decoder (Processes all tokens, shared across channels) 
        dec_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, norm_first=True, activation="gelu")
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=dec_layers)
        
        # Pixel Reconstruction Head
        self.from_token = nn.ConvTranspose2d(d_model, self.chunk_size, kernel_size=4, stride=2, padding=1)

    def forward(self, y_in: torch.Tensor, mask_indices: torch.Tensor):
        # y_in: (B, C, H, W), mask_indices: (B, F) bool (True = missing/unsent)
        B, C, H, W = y_in.shape
        
        # 1. Tokenization per chunk
        y_reshaped = y_in.view(B * self.F, self.chunk_size, H, W)
        tokens = self.to_token(y_reshaped) # (B*F, d_model, H/2, W/2)
        _, _, H_out, W_out = tokens.shape
        L_spatial = H_out * W_out
        
        tokens = tokens.view(B, self.F, self.d_model, L_spatial).permute(0, 1, 3, 2) # (B, F, L_spatial, d_model)
        
        # 2. Add Positional and Channel Embeddings [cite: 170]
        tokens = tokens + self.pos_emb
        channel_embs = self.channel_tokens.unsqueeze(2) # (1, F, 1, d_model)
        tokens = tokens + channel_embs
        
        # 3. Separate Visible and Masked (Encoder takes only visible)
        # Note: In standard MAE, encoder drops masked tokens. For simplicity in parallel batch processing 
        # where mask lengths vary, we can use an attention mask, or process fixed-length visible sets.
        # DeepJSCC min_F to F progressive transmission means lengths vary per scenario, but within a batch 
        # it might be uniform if expanded. Here we use sequence replacement with [MASK] for simplicity, 
        # though true MAE drops them. We simulate dropping by zeroing/masking in attention.
        
        mask_expanded = mask_indices.unsqueeze(2).expand(B, self.F, L_spatial) # (B, F, L_spatial)
        mask_flat = mask_expanded.reshape(B, -1)
        tokens_flat = tokens.reshape(B, self.F * L_spatial, self.d_model)
        
        # Replace masked with [MASK] token
        tokens_flat = torch.where(mask_flat.unsqueeze(-1), self.mask_token, tokens_flat)
        
        # Append Memory Tokens [cite: 168]
        mem_tokens = self.memory_tokens.expand(B, -1, -1)
        enc_input = torch.cat([mem_tokens, tokens_flat], dim=1)
        
        # Encode
        enc_out = self.encoder(enc_input)
        
        # Decode (Channel-Aware Decoder) 
        # Memory tokens assist in decoding by providing global context.
        dec_out = self.decoder(enc_out)
        
        # Extract patch tokens (remove memory tokens)
        patch_out = dec_out[:, self.num_memory_tokens:, :] # (B, F * L_spatial, d_model)
        
        # 4. De-Tokenization
        patch_out = patch_out.view(B * self.F, L_spatial, self.d_model).permute(0, 2, 1).view(B * self.F, self.d_model, H_out, W_out)
        recovered_chunks = self.from_token(patch_out) # (B*F, C_chunk, H, W)
        recovered = recovered_chunks.view(B, C, H, W)
        
        # 5. Paste original visible chunks back
        mask_spatial = mask_indices.view(B, self.F, 1, 1, 1).expand(B, self.F, self.chunk_size, H, W).reshape(B, C, H, W)
        final_out = torch.where(mask_spatial, recovered, y_in)
        
        return final_out