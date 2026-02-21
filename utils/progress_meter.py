import sys
from tqdm import tqdm
import torch

__all__ = ['progressMeter']

class progressMeter:
    def __init__(self, desc, writer, logger, total_data_len, epoch, use_pbar):
        # 텐서 대신 일반 float 누적 딕셔너리 사용
        self.states_sum = {'batch_size': 0.0}
        
        self.writer = writer
        self.logger = logger
        self.use_pbar = use_pbar
        self.epoch = epoch
        self.desc = f'Epoch {epoch} | {desc}:'
        self.total_data_len = total_data_len
        
        if use_pbar:
            # ncols를 제거하고 {bar:15}를 사용하여 막대 길이만 고정
            # 괄호 외부에 {postfix}를 배치하여 긴 지표 문자열이 잘리지 않도록 설정
            custom_format = '{desc} {percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}'
            self.pbar = tqdm(total=total_data_len, file=sys.stdout, desc=self.desc, bar_format=custom_format)
        
    def update(self, state_dict):
        assert 'batch_size' in state_dict.keys(), "Batch size should be provided in states! (as 'batch_size')"
        assert 'loss' in state_dict.keys(), "Loss should be provided in states! (as 'loss')"
        
        self.state_size = state_dict.pop('batch_size')
        self.states_sum['batch_size'] += self.state_size
        
        for name, val in state_dict.items():
            if torch.is_tensor(val):
                val = val.detach().item() # GPU 텐서를 CPU float로 변환
                
            if name not in self.states_sum:
                self.states_sum[name] = 0.0
            self.states_sum[name] += val * self.state_size
    
    def verbose_states(self):
        total_batch = self.states_sum['batch_size']
        avg_states = {name: val / total_batch for name, val in self.states_sum.items() if name != 'batch_size'}
        
        if self.use_pbar:
            self.pbar.update(int(self.state_size))
            
            # 원하는 형식의 문자열로 직접 조립
            out_str = ""
            for name, val in avg_states.items():
                if name=='PSNR':
                    out_str += f" | {name}:{val:.2f}"
                else:
                    out_str += f" | {name}:{val:.5f}"
                
            # 조립된 문자열을 postfix로 전달
            self.pbar.set_postfix_str(out_str[1:])
        else:
            out = f"{self.desc}"
            for name, val in avg_states.items():
                if name=='PSNR':
                    out_str += f" | {name}:{val:.2f}"
                else:
                    out_str += f" | {name}:{val:.5f}"
            print(out[1:])
    
    def write_summary(self):
        total_batch = self.states_sum['batch_size']
        avg_states = {name: val / total_batch for name, val in self.states_sum.items() if name != 'batch_size'}
        
        out = ''
        for name, val in avg_states.items():
            self.writer.add_scalar(name, val, self.epoch)
            out += f"|{name}: {val:.5f}"
            
        epoch_prog = f"({int(total_batch):4d}/{self.total_data_len:4d})"
        self.logger.debug(f'Epoch Summary|{self.desc} {epoch_prog}{out[1:]}')
        
        if self.use_pbar:
            self.pbar.close()
            
        return avg_states['loss'] # Average Loss