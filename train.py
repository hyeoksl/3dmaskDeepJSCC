import os
import sys
import random
import yaml
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torchinfo import summary

import model as jscc_models
from utils import load_dataloaders, getUsableGPUs,\
    progressMeter, logger_configuration, load_ckpt, save_ckpt, custom_arg_parsing, QuadResJSCCLoss
from setproctitle import setproctitle

def train_epoch(epoch, dataloader, model, criterion, clip_max_norm, optimizer, writer, logger):
    model.train()    
    total_images = len(dataloader.dataset)
    progress = progressMeter('train', writer, logger, total_images, epoch, use_pbar=True)
    
    for data in dataloader:
        data = data.cuda()
        
        optimizer.zero_grad()
        out_net = model(data)
        loss_dict = criterion(out_net, data)
        loss_dict['loss'].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
    
        loss_dict['PSNR'] = 10 * torch.log10(1/loss_dict['loss'])
        loss_dict['batch_size'] = len(data)
        # Needed for overall curves in ResJSCC
        
        progress.update(loss_dict)
        progress.verbose_states()
    
        
    progress.write_summary()
    
def test_epoch(epoch, dataloader, model, criterion, writer, logger):
    model.eval()
    total_images = len(dataloader.dataset)
    
    progress = progressMeter('valid', writer, logger, total_images, epoch, use_pbar=True)
    
    with torch.no_grad():
        for data in dataloader:
            data = data.cuda()
            
            out_net = model(data)
            loss_dict = criterion(out_net, data, training=False)
            
            loss_dict['PSNR'] = 10 * torch.log10(1/loss_dict['loss'])
            loss_dict['batch_size'] = len(data)
            
            progress.update(loss_dict)
            progress.verbose_states()
    
    # 이 rate의 평균 loss 기록
    rate_avg_loss = progress.write_summary()
    
    return rate_avg_loss


def main(argv):
    # Argument parsing
    config = custom_arg_parsing(argv)
    
    # GPU assign
    if not config.use_cpu and not config._gpu_assigned:
        gpu_id = getUsableGPUs()
        if gpu_id is None:
            print("No GPUs Available!")
            return
        torch.cuda.set_device(gpu_id)
    else:
        if torch.cuda.is_available():
            print("Using CUDA")
            torch.cuda.set_device(0)
    
    # 자동으로 history.txt에 모든 명령어 기록 저장.
    logfile = "history.txt"

    with open(logfile, "a") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] python ")
        f.write(" ".join(sys.argv))
        f.write("\n")
        
    # Seed  
    if config.seed is not None:
        torch.manual_seed(config.seed)
        random.seed(config.seed)
    
    if config.logging.save_dir!='logs/':
        setproctitle(config.logging.save_dir[5:]+'_'+config.experiment_name)
    else:
        setproctitle(config.experiment_name)
    # Logging
    cfg_log = config.logging
    logger = logger_configuration(cfg_log, config.experiment_name)
    logger.info(config)
    with open(f'{cfg_log.save_path}/config.yaml', 'w') as f:
        yaml.dump(config, f)
    train_writer = SummaryWriter(os.path.join(cfg_log.save_path, "train"))
    valid_writer = SummaryWriter(os.path.join(cfg_log.save_path, "valid"))

    # Model
    cfg_model = config.model
    print(cfg_model)
    model_class = getattr(jscc_models, cfg_model.type)
    del cfg_model.type
    model = model_class(**cfg_model).cuda()
    H, W = cfg_model.img_shape
    logger.info('SUMMARY:\n'+str(summary(model, (1,3,H,W), depth=2,device='cuda', verbose=0)))

    # Criterion
    criterion = QuadResJSCCLoss(model=model,loss_type=config.loss_type)
    
    # Dataset
    cfg_dataset = config.dataset
    train_dl, valid_dl = load_dataloaders(**cfg_dataset)
    logger.info(f"Number of dataset | train : {len(train_dl.dataset)}, valid : {len(valid_dl.dataset)}")
    
    # Optimizer
    cfg_optim = config.optimizer
    if cfg_optim.type == 'Adam':        
        optimizer = optim.Adam(model.parameters(), lr=cfg_optim.learning_rate)
    else:
        raise NameError(f"optimizer.type {cfg_optim.type} is not Supported!")
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, cfg_optim.lr_milestone_epoch, \
        gamma=cfg_optim.lr_gamma, last_epoch=-1)
    
    # Load Checkpoint 
    start_epoch = load_ckpt(cfg_log.pretrain_ckpt, model, optimizer, lr_scheduler, logger)
    
    # Start training!
    best_loss = float("inf")
    old_lr = 'None'
    for epoch in range(start_epoch, config.epochs):
        # Print learning rate update
        now_lr = optimizer.param_groups[0]['lr']
        if old_lr != now_lr:
            logger.info(f"Learning rate change: {old_lr} -> {now_lr}")
            old_lr = now_lr
    
        train_epoch(
            epoch,
            train_dl, 
            model, 
            criterion, 
            config.clip_max_norm, 
            optimizer, 
            train_writer, 
            logger)
        loss = test_epoch(
            epoch,
            valid_dl,
            model,
            criterion,
            valid_writer,
            logger
            )
        lr_scheduler.step()

        # Check best
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
    
        # Save CKPT
        if cfg_log.save_model:
            save_ckpt(
                cfg_log,
                epoch, 
                loss,  
                is_best,
                model,
                optimizer,
                lr_scheduler,
                logger
            )

if __name__ == "__main__":
    main(sys.argv[1:])
