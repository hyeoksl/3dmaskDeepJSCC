import os
import logging
import torch

__all__ = ['logger_configuration', 'load_ckpt', 'save_ckpt']

def logger_configuration(config, experiment_name):
    logger = logging.getLogger("DeepJSCC")
    config.save_path = os.path.join(config.save_dir, experiment_name)
    log_path = config.save_path + '/logging.log'
        
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path, exist_ok=True)
    if config.save_every > 0:
        weight_path = os.path.join(config.save_path, "weights")
        if not os.path.exists(weight_path):
            os.makedirs(weight_path, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                "%Y-%m-%d %H:%M:%S")
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    filehandler = logging.FileHandler(log_path)
    filehandler.setLevel(logging.DEBUG)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    logger.setLevel(logging.DEBUG)
    
    return logger


def load_ckpt(ckpt, model, optimizer, lr_scheduler, logger):
    if ckpt is not None:
        try:
            checkpoint = torch.load(ckpt, 
                                    map_location=torch.device('cuda'))
            model.load_state_dict(checkpoint["state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            logger.info("Loaded from" + ckpt)
        except:
            start_epoch = 0
            logger.info("Failed to load state from" + ckpt + "Training from scratch...")
    else:
        start_epoch = 0
        logger.info("Training from scratch...")
        
    return start_epoch

def save_ckpt(config, epoch, loss, is_best, model, optimizer, lr_scheduler, logger):
    state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "loss": loss,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
    }
    # Save latest
    torch.save(state, os.path.join(config.save_path, f"latest_ckpt_ep{epoch}.pth.tar"))
    for fname in os.listdir(config.save_path):
        if 'latest_ckpt' in fname and fname != f"latest_ckpt_ep{epoch}.pth.tar":
            os.remove(os.path.join(config.save_path, fname))
    logger.info(f"Saved latest model in epoch : {epoch}")
    # Save every
    if (epoch+1) % config.save_every == 0:
        torch.save(state, os.path.join(config.save_path, 'weights', f"Epoch_{epoch}.pth.tar"))
        logger.info(f"Saved model milestone in epoch : {epoch}")
    # Save best
    if is_best:
        torch.save(state, os.path.join(config.save_path, f"best_ckpt_ep{epoch}.pth.tar"))
        for fname in os.listdir(config.save_path):
            if 'best_ckpt' in fname and fname != f"best_ckpt_ep{epoch}.pth.tar":
                os.remove(os.path.join(config.save_path, fname))
        logger.info(f"Saved best model in epoch : {epoch}")
    