
import os
import torch
import torch.nn as nn

class CustomJSCCLoss(nn.Module):
    """Custom MSE loss"""

    def __init__(self, loss_type='MSE'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.loss_type = loss_type
            
    def forward(self, output, target, training=True):
        out = {}
        out["MSE"] = self.mse(output["x_hat"], target)
        
        if self.loss_type == 'MSE':
            out["loss"] = out["MSE"]
        else:
            raise NameError(f"criterion_type {self.loss_type} is not Supported!")

        return out
    
class QuadResJSCCLoss(nn.Module):
    """Custom MSE loss for ResJSCC"""
    def __init__(self, model, loss_type='MSE'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.loss_type = loss_type
        self.weights=torch.tensor([i**2 for i in range(model.min_chunks, model.F+1)], device='cuda', dtype=torch.float32)
        self.weights=self.weights/self.weights.mean(dim=0)
            
    def forward(self, output, target, training=True):
        out={}
        x_hats = output["x_hat"]                 # (B, 3, H, W) F개
        x_hat=torch.stack(x_hats, dim=1)
        B, F, C, H, W = x_hat.shape
        
        # --- expand GT to compare with every accumulated image -------------
        target_rep = target.unsqueeze(1).expand(-1, F, -1, -1, -1)  # (B, F, 3, H, W)

        # --- per-step MSE: average over C, H, W, then batch ---------------
        mse_curve = torch.mean((x_hat - target_rep) ** 2, dim=(0,2, 3, 4))   # (F)
        # --- Loss: only the first reconstruction drives back-prop ---------
        if self.loss_type == 'MSE':
            out['loss'] = torch.mean(self.weights*mse_curve)
        else:
            raise NameError(f"criterion_type {self.loss_type} is not Supported!")
        out["MSE"]=mse_curve[-1].detach()
        for i in range(0, F, max(F//4,1)):
            out[f"MSE(~{i+1}th)"]=mse_curve[i].detach()
        
        return out
    