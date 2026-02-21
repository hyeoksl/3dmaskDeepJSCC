import torch
import torch.nn as nn

from einops import rearrange
from compressai.models.utils import conv, deconv

from model.layers import *
import math, os, yaml

CHUNK_SIZE=32
MIN_SPP=0.1

__all__ = [
    "SemCNN_lite",
    "Padding",
    "GroupRecover",
    "Conv3dRecover",
    "ChunkGITJSCC",
    "BERT3D"
]
class SemCNN_lite(nn.Module):
    def __init__(self, SNRdB=10, spp=0.25, img_shape=(256, 256), dim=384, **kwargs):
        super().__init__()
        self.ds_factor = 2 ** 4
        symbol_dim = (int(spp * (self.ds_factor**2) * 2)//CHUNK_SIZE)*CHUNK_SIZE # Real and Imag : *2
        self.F=symbol_dim//CHUNK_SIZE
        self.min_F=math.ceil(int(MIN_SPP * (self.ds_factor**2) * 2)/CHUNK_SIZE)
        self.snr=SNRdB
        
        self.encoder = nn.Sequential(
            conv(3, dim,3),
            resNetBlock(dim),
            conv(dim, 3*dim//2,3),
            resNetBlock(3*dim//2),
            conv(3*dim//2, 2*dim,3),
            resNetBlock(2*dim),
            conv(2*dim, symbol_dim,3),
        )
        self.decoder = nn.Sequential(
            deconv(symbol_dim, 2*dim,3),
            resNetBlock(2*dim),
            deconv(2*dim, 3*dim//2,3),
            resNetBlock(3*dim//2),
            deconv(3*dim//2, dim,3),
            resNetBlock(dim),
            deconv(dim, 3,3),
        )
        self.channel = AWGNChannel(SNRdB=SNRdB)

    def get_parser(parser):
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dim', type=int)
        parser.add_argument('--model.img_shape', type=tuple)
        
        return parser
    
    def forward(self, x):
        y = self.encoder(x) # (B, symbol_dim=128, 16, 16)
        B,C,H,W=y.shape
        y = rearrange(y, "b (n_c iq) h w -> b (n_c h w) iq", iq=2) # (B, 64, 16*16, 2)
        y = torch.complex(y[..., 0], y[..., 1])
        y=y/torch.sqrt(torch.mean(torch.abs(y)**2, dim=1, keepdim=True)+1e-6)  
        
        if self.training:instsnr=torch.randint(-5, 25, (B, 1), device=x.device).float()
        else: instsnr=torch.ones(B, 1, device=x.device) * self.snr
        y_hat = self.channel(y, snr_db=instsnr)
        
        y_hat = torch.stack((y_hat.real, y_hat.imag), dim=-1) # (B, 64, 16*16, 2)
        y_hat = rearrange(y_hat, "b (n_c h w) iq -> b (n_c iq) h w", h=H, w=W) # (B, 128, 16, 16)
        x_hat = self.decoder(y_hat) # (B, 3, 256, 256)

        x_hat = torch.clamp(x_hat, 0, 1)

        # return x_hat
        return {
            "x_hat": x_hat, "y":y
        }

class Padding(nn.Module):
    """
    Kurka, David Burth, and Deniz Gündüz. "Successive refinement of images with deep joint source-channel coding." 2019 IEEE 20th International Workshop on Signal Processing Advances in Wireless Communications (SPAWC). IEEE, 2019.
    
    Single encoder-decoder model. This is GOAT in rate adaptiveness
    """
    def __init__(self, SNRdB=10, spp=0.25, img_shape=(256, 256), dim=384, **kwargs):
        super().__init__()
        # 일단 학습을 빠르게 돌리기 위해 작은 모델로 진행.
        self.ds_factor = 2 ** 4
        symbol_dim = (int(spp * (self.ds_factor**2) * 2)//CHUNK_SIZE)*CHUNK_SIZE # Real and Imag : *2
        self.F=symbol_dim//CHUNK_SIZE
        self.min_F=math.ceil(int(MIN_SPP * (self.ds_factor**2) * 2)/CHUNK_SIZE)
        self.snr=SNRdB
        
        self.encoder = nn.Sequential(
            conv(3, dim,3),
            resNetBlock(dim),
            conv(dim, 3*dim//2,3),
            resNetBlock(3*dim//2),
            conv(3*dim//2, 2*dim,3),
            resNetBlock(2*dim),
            conv(2*dim, symbol_dim,3),
        )
        self.decoder = nn.Sequential(
            deconv(symbol_dim, 2*dim,3),
            resNetBlock(2*dim),
            deconv(2*dim, 3*dim//2,3),
            resNetBlock(3*dim//2),
            deconv(3*dim//2, dim,3),
            resNetBlock(dim),
            deconv(dim, 3,3),
        )
        self.channel = AWGNChannel(SNRdB=SNRdB)   
        
    def get_parser(parser):
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dim', type=int)
        parser.add_argument('--model.img_shape', type=tuple)
        
        return parser
    
    def forward(self, x):

        y = self.encoder(x) # (B, symbol_dim=128, 16, 16)
        B,C,H,W=y.shape
        y = rearrange(y, "b (n_c iq) h w -> b (n_c h w) iq", iq=2) # (B, 64, 16*16, 2)
        y = torch.complex(y[..., 0], y[..., 1])
        y=y/torch.sqrt(torch.mean(torch.abs(y)**2, dim=1, keepdim=True)+1e-6)
        
        if self.training:instsnr=torch.randint(-5, 25, (B, 1), device=x.device).float()
        else: instsnr=torch.ones(B, 1, device=x.device) * self.snr
        y_hat = self.channel(y, snr_db=instsnr)
        
        y_hat = torch.stack((y_hat.real, y_hat.imag), dim=-1) # (B, 64, 16*16, 2)
        y_hat = rearrange(y_hat, "b (n_c h w) iq -> b (n_c iq) h w", h=H, w=W) # (B, 128, 16, 16)
        
        y_hat = rearrange(y_hat, "b (f c_new) h w -> b f c_new h w", c_new=CHUNK_SIZE)
        zero_padding=torch.zeros_like(y_hat)

        scenarios=[]
        for i in range(self.min_F,self.F+1):
            scenarios.append(torch.cat((y_hat[:,:i], zero_padding[:,i:]), dim=1))
        y_hat_diff=torch.stack(scenarios,dim=1)
        y_hat=rearrange(y_hat_diff, "b s f c h w -> b s (f c) h w")
        y_hat=rearrange(y_hat, "b s c h w -> (b s) c h w")
        y_hat = self.decoder(y_hat) # (B*F, 3, 256, 256)
        y_hat = rearrange(y_hat, "(b f) c h w -> b f c h w", b=B)

        y_hat = torch.clamp(y_hat, 0, 1)
        
        return {
            "x_hat": y_hat, 
            "y": y, 
        }
    
class GroupRecover(nn.Module):
    """
    위에 WaveNet은 dilated Causal convolution인데 이거 그냥 grouped convolution 쓰면 되는 거 아닌가 해서 만든 거임.
    Channel을 앞에서부터 남기는 Padding이냐, 아니면 그냥 뽑고 랜덤하게 날릴까 했는데 Masked Autoencoder라면 랜덤하게가 나을 것 같음.
    """
    def __init__(self, SNRdB=10, log_dir='logs/Baseline/10dB_o5_192', **kwargs):
        super().__init__()
        
        # Load Pretrained JSCC model
        with open(os.path.join(log_dir, 'config.yaml')) as f:
            config = yaml.unsafe_load(f)
        model_cfg = config.model
        model_type=model_cfg.type
        del model_cfg.type
        if model_type=='SemCNN_lite':
            self.JSCC=SemCNN_lite(**model_cfg).cuda()
        elif model_type=='Padding':
            self.JSCC=Padding(**model_cfg).cuda()
        else:
            raise NameError(f"{model_type} not supported!")
        
        for item in os.listdir(log_dir):
            if 'best_ckpt' in item:
                ckpt = os.path.join(log_dir, item)
                break
    
        checkpoint = torch.load(ckpt, map_location=torch.device('cuda'))
        self.JSCC.load_state_dict(checkpoint["state_dict"])
        self.JSCC.training=True
        self.snr=SNRdB
        print("Loaded from" + ckpt)

        self.ds_factor = self.JSCC.ds_factor
        self.F=self.JSCC.F
        self.min_F=self.JSCC.min_F
        symbol_dim=self.F*CHUNK_SIZE
        
        steps=math.ceil(math.log2(self.F))
        print(steps)

        print(self.min_F, self.F)

        temp=[]
        for i in range(steps-1, -1, -1):
            temp.append(nn.Conv2d(symbol_dim,symbol_dim,3,padding=1,groups=2**i))
            temp.append(resNetBlock(symbol_dim,groups=2**i))
        self.filler=nn.Sequential(
            *temp
        )

        self.channel = AWGNChannel(SNRdB=SNRdB)

    def forward(self, x):
        # 1. Encoder
        y = self.JSCC.encoder(x)
        B,C,H,W=y.shape
        y = rearrange(y, "b (n_c iq) h w -> b (n_c h w) iq", iq=2) # (B, 64, 16*16, 2)
        y = torch.complex(y[..., 0], y[..., 1])
        y=y/torch.sqrt(torch.mean(torch.abs(y)**2, dim=1, keepdim=True)+1e-6)
        # 2. Channel
        if self.training:instsnr=torch.randint(-5, 25, (B, 1), device=x.device).float()
        else: instsnr=torch.ones(B, 1, device=x.device) * self.snr
        y_hat = self.channel(y, snr_db=instsnr)
        
        y_hat = torch.stack((y_hat.real, y_hat.imag), dim=-1) # (B, 64, 16*16, 2)
        y_hat = rearrange(y_hat, "b (n_c h w) iq -> b (n_c iq) h w", h=H, w=W) # (B, 128, 16, 16)
        # 3. Decoder
        # 3-1. Get Masked Sequence
        ys = []
        for s in range(self.min_F, self.F + 1):
            k = s*CHUNK_SIZE  # 남길 channel 수
            
            # (1) 채널 인덱스 샘플링: [0..C-1] 랜덤 permutation에서 앞의 k개 사용
            keep_idx = torch.randperm(C, device=y_hat.device)[:k]  # (k,)  :contentReference[oaicite:1]{index=1}

            # (2) 채널 마스크 생성 후 적용 (남길 채널만 1)
            ch_mask = torch.zeros(C, device=y_hat.device, dtype=y_hat.dtype)  # (C,)
            ch_mask[keep_idx] = 1.0
            current_y = y_hat * ch_mask.view(1, C, 1, 1)  # broadcast to (B,C,H,W)

            ys.append(current_y)
        y_hats=torch.cat(ys, dim=0)

        # 3-2. Filler pass
        refined=self.filler(y_hats)
        refined=rearrange(refined, '(b f) c h w -> b f c h w',b=B)
        for i in range(refined.shape[1]):
            refined[:,i]=torch.cat((y_hat[:,:CHUNK_SIZE*(i+self.min_F)],refined[:,i,CHUNK_SIZE*(self.min_F+i):]),dim=1)
        refined=rearrange(refined, 'b s c h w -> (b s) c h w')

        # 3-3. Decoder pass
        compensated_all = self.JSCC.decoder(refined)
        compensated_all = torch.clamp(compensated_all, 0, 1)
        # Reshape to (B, F, 3, H, W) for comparison
        x_hat = rearrange(compensated_all, '(b s) c h w -> b s c h w', b=B)

        return {
            "x_hat": x_hat,           # (B, 8, 3, H, W) -> The filled/reproduced versions
            "y": y,
        }
    def get_parser(parser):
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dim', type=int)
        parser.add_argument('--model.img_shape', type=tuple)
        
        return parser

class Conv3dRecover(nn.Module):
    """
    (B,C,H,W)를 (B,chunk_size,T=8,H,W)로 쪼개 3d convolution을 이용하여 복원하겠다는 생각.
    grouped convolution과 다른 점? grouped convolution은 이거로 치면 T쪽 stride가 T kernel size랑 같은 거라 ㅇㅇ.
    위와 마찬가지로 random하게 날려야겠지만, 얘는 positional embedding이 필요할 듯?
    """
    def __init__(self, SNRdB=10, kernel_size=3, log_dir='logs/Baseline/10dB_o5_192', **kwargs):
        super().__init__()
        
        # Load Pretrained JSCC model
        with open(os.path.join(log_dir, 'config.yaml')) as f:
            config = yaml.unsafe_load(f)
        model_cfg = config.model
        model_type=model_cfg.type
        del model_cfg.type
        if model_type=='SemCNN_lite':
            self.JSCC=SemCNN_lite(**model_cfg).cuda()
        elif model_type=='Padding':
            self.JSCC=Padding(**model_cfg).cuda()
        else:
            raise NameError(f"{model_type} not supported!")
        
        for item in os.listdir(log_dir):
            if 'best_ckpt' in item:
                ckpt = os.path.join(log_dir, item)
                checkpoint = torch.load(ckpt, map_location=torch.device('cuda'))
                self.JSCC.load_state_dict(checkpoint["state_dict"])
                break
    
        self.JSCC.training=True
        self.snr=SNRdB
        print("Loaded from" + ckpt)

        self.ds_factor = self.JSCC.ds_factor
        self.F=self.JSCC.F
        self.min_F=self.JSCC.min_F
        symbol_dim=self.F*CHUNK_SIZE

        print(self.min_F, self.F)

        # 한 step마다 kernel_size//2*2 만큼 까이니까 8 chunk가 1 symbol이 되려면 4step이 드네.

        self.kernel_size=max(kernel_size,self.min_F)
        steps=math.ceil(self.F/(self.kernel_size//2*2))  # 아마 4가 나올 거임. self.F가 8이고 self.kernel_size=3일 테니까. 만약 self.F가 8이 아니라 16이라면, self.min_F는 2니까 뭐...

        self.length=1+steps*(self.kernel_size//2*2)  # 이러면 9개임.
        
        self.filler = WaveNet3D(
            dims=[CHUNK_SIZE*2**i for i in range(1,steps)]+[symbol_dim],
            kernel_size=self.kernel_size,
            chunk_dim=CHUNK_SIZE,
            input_length=self.length,
            num_blocks=4
        )
        self.channel = AWGNChannel(SNRdB=SNRdB)

    def forward(self, x):

        y = self.JSCC.encoder(x)
        B,C,H,W=y.shape
        y = rearrange(y, "b (n_c iq) h w -> b (n_c h w) iq", iq=2) # (B, 64, 16*16, 2)
        y = torch.complex(y[..., 0], y[..., 1])
        y=y/torch.sqrt(torch.mean(torch.abs(y)**2, dim=1, keepdim=True)+1e-6)
        
        if self.training:instsnr=torch.randint(-5, 25, (B, 1), device=x.device).float()
        else: instsnr=torch.ones(B, 1, device=x.device) * self.snr
        y_hat = self.channel(y, snr_db=instsnr)
        
        y_hat = torch.stack((y_hat.real, y_hat.imag), dim=-1) # (B, 64, 16*16, 2)
        y_hat = rearrange(y_hat, "b (n_c h w) iq -> b (n_c iq) h w", h=H, w=W) # (B, 128, 16, 16)
        
        # 1. Prepare Chunked Sequence
        # Shape: (B, F, C_chunk, H, W)
        y_hat_chunked = rearrange(y_hat, 'b (f c) h w -> b f c h w', c=CHUNK_SIZE)
        ys = []
        for s in range(self.min_F, self.F + 1):
            k = s  # 남길 chunk 수
            
            # (1) 채널 인덱스 샘플링: [0..C-1] 랜덤 permutation에서 앞의 k개 사용
            keep_idx = torch.randperm(self.F, device=y_hat.device)[:k]  # (k,)  :contentReference[oaicite:1]{index=1}

            # (2) 채널 마스크 생성 후 적용 (남길 채널만 1)
            ch_mask = torch.zeros(self.F, device=y_hat.device, dtype=y_hat.dtype)  # (C,)
            ch_mask[keep_idx] = 1.0
            current_y = y_hat_chunked * ch_mask.view(1, self.F,1, 1, 1)  # broadcast to (B,C,H,W)

            ys.append(current_y)
        y_hats_chunked=torch.cat(ys, dim=0)

        # 2. Filler pass
        # Every index 't' in y_long refined based on its past

        refined=self.filler(y_hats_chunked)

        compensated_all = self.JSCC.decoder(refined)
        compensated_all = torch.clamp(compensated_all, 0, 1)
        # Reshape to (B, F, 3, H, W) for comparison
        x_hat = rearrange(compensated_all, '(b s) c h w -> b s c h w', b=B)

        return {
            "x_hat": x_hat,           # (B, 8, 3, H, W) -> The filled/reproduced versions
            "y": y,
        }
    def get_parser(parser):
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dim', type=int)
        parser.add_argument('--model.img_shape', type=tuple)
        
        return parser
        
class ChunkGITJSCC(nn.Module):
    def __init__(self, SNRdB=10, spp=0.5, img_shape=(256, 256), dim=192, 
                 log_dir='logs/Baseline/10dB_o5_192', 
                 topk_ratio=0.3, # Mixed Strategy Ratio
                 **kwargs):
        super().__init__()
        # Load Pretrained JSCC model
        with open(os.path.join(log_dir, 'config.yaml')) as f:
            config = yaml.unsafe_load(f)
        model_cfg = config.model
        model_type=model_cfg.type
        del model_cfg.type
        if model_type=='SemCNN_lite':
            self.JSCC=SemCNN_lite(**model_cfg).cuda()
        elif model_type=='Padding':
            self.JSCC=Padding(**model_cfg).cuda()
        else:
            raise NameError(f"{model_type} not supported!")
        
        for item in os.listdir(log_dir):
            if 'best_ckpt' in item:
                ckpt = os.path.join(log_dir, item)
                break
    
        checkpoint = torch.load(ckpt, map_location=torch.device('cuda'))
        self.JSCC.load_state_dict(checkpoint["state_dict"])
        self.JSCC.training=True
        self.snr=SNRdB
        print("Loaded from " + ckpt)

        self.ds_factor = self.JSCC.ds_factor
        self.F=self.JSCC.F
        self.CHUNK_SIZE = self.JSCC.C // self.F if hasattr(self.JSCC, 'C') else config.model.C // self.F # 명시적 chunk size 계산
        symbol_dim = self.F * self.CHUNK_SIZE # Real and Imag : *2 (기존 C와 동일해야 함)
        self.symbol_dim = symbol_dim
        self.min_F=self.JSCC.min_F

        self.topk_ratio=topk_ratio

        # Filler (MaskGITbutChannel)
        self.filler = ChunkMaskGIT(
            symbol_dim=symbol_dim,    # C
            d_model=4*symbol_dim,
            nhead=8,
            num_layers=4,
            dropout=0.1,
            num_steps=6,
            len=self.F
        )
        self.channel = AWGNChannel(SNRdB=SNRdB)
    
    def get_energy_based_indices(self, y, k, largest=True):
        # (B, C, H, W) -> (B, F, C//F, H, W) -> Energy per chunk -> Top-k indices
        B, C, H, W = y.shape
        chunk_size = C // self.F
        
        y_reshaped = y.view(B, self.F, chunk_size, H, W)
        energy = torch.mean(y_reshaped**2, dim=(2, 3, 4)) 
        indices = torch.topk(energy, k=k, dim=1, largest=largest).indices
        return indices

    def forward(self, x):
        # 1. JSCC Encode & Channel Simulation
        y = self.JSCC.encoder(x) 
        B, C, H, W = y.shape
        
        # (기존 JSCC 통신 채널 모사 로직 유지)
        y = rearrange(y, "b (n_c iq) h w -> b (n_c h w) iq", iq=2)
        y = torch.complex(y[..., 0], y[..., 1])
        y = y / torch.sqrt(torch.mean(torch.abs(y)**2, dim=1, keepdim=True) + 1e-6)
        
        if self.training:
            instsnr = torch.randint(-5, 25, (B, 1), device=x.device).float()
        else: 
            instsnr = torch.ones(B, 1, device=x.device) * self.snr
            
        y_hat = self.channel(y, snr_db=instsnr)
        
        y_hat = torch.stack((y_hat.real, y_hat.imag), dim=-1)
        y_hat = rearrange(y_hat, "b (n_c h w) iq -> b (n_c iq) h w", h=H, w=W)
        
        # Shape: (B, F, C_chunk, H, W)
        y_hat_chunked = rearrange(y_hat, 'b (f c) h w -> b f c h w', c=CHUNK_SIZE)
        ys = []
        masks=[]
        for s in range(self.min_F, self.F + 1):
            k = s  # 남길 chunk 수
            
            # (1) 채널 인덱스 샘플링: [0..C-1] 랜덤 permutation에서 앞의 k개 사용
            keep_idx = torch.randperm(self.F, device=y_hat.device)[:k]  # (k,)  :contentReference[oaicite:1]{index=1}

            # (2) 채널 마스크 생성 후 적용 (남길 채널만 1)
            ch_mask = torch.zeros(self.F, device=y_hat.device, dtype=y_hat.dtype)  # (C,)
            ch_mask[keep_idx] = 1.0
            current_y = y_hat_chunked * ch_mask.view(1, self.F,1, 1, 1)  # broadcast to (B,C,H,W)

            ys.append(current_y)
            mask = torch.ones((B, self.F), dtype=torch.bool, device=y.device)
            mask.scatter_(1, keep_idx, False) 
            masks.append(mask)
        y_hats_chunked=torch.cat(ys, dim=0)      
        masks_stack = torch.stack(masks, dim=1)   # (B, 8, F)

        # 2. Filler pass
        # Every index 't' in y_long refined based on its past  
        y_in = rearrange(y_hats_chunked, 'b s c h w -> b (s c) h w')
        mask_in = rearrange(masks_stack, 'b s f -> (b s) f')

        # Filler Inference
        predicted_chunks = self.filler(y_in, mask_indices=mask_in)

        # JSCC Decode
        x_hat = self.JSCC.decoder(predicted_chunks)
        x_hat = torch.clamp(x_hat, 0, 1)
        
        x_hat = rearrange(x_hat, '(b s) c h w -> b s c h w', b=B)
        
        return {
            "x_hat": x_hat, 
            "y": y,
        }
    def get_parser(parser):
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dim', type=int)
        parser.add_argument('--model.img_shape', type=tuple)
        
        return parser
    
class BERT3D(nn.Module):
    def __init__(self, SNRdB=10, log_dir='logs/Baseline/10dB_o5_192', CHUNK_SIZE=16, **kwargs):
        super().__init__()
        
        with open(os.path.join(log_dir, 'config.yaml')) as f:
            config = yaml.unsafe_load(f)
        model_cfg = config.model
        model_type = model_cfg.type
        del model_cfg.type
        
        if model_type == 'SemCNN_lite':
            self.JSCC = SemCNN_lite(**model_cfg).cuda()
        elif model_type == 'Padding':
            self.JSCC = Padding(**model_cfg).cuda()
        else:
            raise NameError(f"{model_type} not supported!")
        
        ckpt = None
        for item in os.listdir(log_dir):
            if 'best_ckpt' in item:
                ckpt = os.path.join(log_dir, item)
                checkpoint = torch.load(ckpt, map_location=torch.device('cuda'))
                self.JSCC.load_state_dict(checkpoint["state_dict"])
                break
    
        self.JSCC.training = True
        self.snr = SNRdB
        print("Loaded from " + str(ckpt))

        self.ds_factor = self.JSCC.ds_factor
        self.F = self.JSCC.F
        self.CHUNK_SIZE = CHUNK_SIZE
        symbol_dim = self.F * self.CHUNK_SIZE # Real and Imag : *2 if needed
        self.min_F = self.JSCC.min_F
        
        # 3D Region-based Filler (ChA-MAEViT)
        self.filler = ChAMAEViT(
            symbol_dim=symbol_dim, 
            F=self.F, 
            d_model=256, 
            nhead=8
        )
        
        self.channel = AWGNChannel(SNRdB=SNRdB)
    
    def forward(self, x):
        # JSCC Encode
        y = self.JSCC.encoder(x)
        B, C, H, W = y.shape
        
        y = rearrange(y, "b (n_c iq) h w -> b (n_c h w) iq", iq=2) 
        y = torch.complex(y[..., 0], y[..., 1])
        y = y / torch.sqrt(torch.mean(torch.abs(y)**2, dim=1, keepdim=True) + 1e-6)

        # Channel Transmission        
        if self.training:
            instsnr = torch.randint(-5, 25, (B, 1), device=x.device).float()
        else: 
            instsnr = torch.ones(B, 1, device=x.device) * self.snr
            
        y_hat = self.channel(y, snr_db=instsnr)
        
        y_hat = torch.stack((y_hat.real, y_hat.imag), dim=-1) 
        y_hat = rearrange(y_hat, "b (n_c h w) iq -> b (n_c iq) h w", h=H, w=W) 
        
        C_chunk = C // self.F
        y_hat_reshaped = y_hat.view(B, self.F, C_chunk, H, W)
        
        y_hats_list = []
        masks_list = []
        
        # 1. Prepare masked versions of y_hat.
        # Simulate progressive JSCC transmission: send i chunks, drop (mask) the rest.
        for i in range(self.min_F, self.F + 1):
            # Mask format: True = missing (unsent), False = visible (sent)
            mask = torch.ones((B, self.F), dtype=torch.bool, device=y_hat.device)
            mask[:, :i] = False  # First i chunks are transmitted
            
            mask_expanded = mask.view(B, self.F, 1, 1, 1).expand(B, self.F, C_chunk, H, W)
            zeros = torch.zeros_like(y_hat_reshaped)
            current_y = torch.where(mask_expanded, zeros, y_hat_reshaped)
            
            y_hats_list.append(current_y.view(B, C, H, W))
            masks_list.append(mask)
            
        # 2. Recover Unsent(masked) coordinates using self.filler
        num_versions = self.F - self.min_F + 1
        
        y_hats_stack = torch.stack(y_hats_list, dim=1) # (B, num_versions, C, H, W)
        masks_stack = torch.stack(masks_list, dim=1)   # (B, num_versions, F)
        
        y_in = rearrange(y_hats_stack, 'b s c h w -> (b s) c h w')
        mask_in = rearrange(masks_stack, 'b s f -> (b s) f')
        
        # Forward through ChA-MAEViT Filler
        refined = self.filler(y_in, mask_indices=mask_in)
        
        # JSCC Decode
        compensated_all = self.JSCC.decoder(refined)
        compensated_all = torch.clamp(compensated_all, 0, 1)
        
        # Reshape to (B, s, C_out, H, W) for comparison
        x_hat = rearrange(compensated_all, '(b s) c h w -> b s c h w', b=B, s=num_versions)

        return {
            "x_hat": x_hat,           # (B, 7, 3, H, W) 
            "y": y,
        }

    @staticmethod
    def get_parser(parser):
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dim', type=int)
        parser.add_argument('--model.img_shape', type=tuple)
        return parser