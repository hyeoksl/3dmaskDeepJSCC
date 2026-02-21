import argparse
import yaml
import model
from jsonargparse import ArgumentParser, ActionConfigFile

__all__ = ['custom_arg_parsing']

def custom_arg_parsing(argv):
    mdl_parser = argparse.ArgumentParser()
    mdl_parser.add_argument('--config_file', type=str, help='path to the yaml file containing params')
    mdl_parser.add_argument('--model.type', type=str, help='model type')
    args, _ = mdl_parser.parse_known_args(argv)
    with open(args.config_file) as f:
        args_dict = yaml.safe_load(f)
    try:
        model_type = args.model.type
    except:
        model_type = None
    if model_type is None:
        try:
            model_type = args_dict['model']['type']
        except:
            model_type = None
    if model_type is None:
        raise AssertionError("Argument model.type should be sepcified!!")
    try:
        model_class = getattr(model, model_type)
    except:
        raise NameError(f"Model type '{model_class}' is invalid! (Not implemented)")
    
    parser = ArgumentParser()
    parser.add_argument('--config_file', action=ActionConfigFile, help='path to the yaml file containing params')
    parser.add_argument('--use_cpu', action='store_true', help='Not to use GPU')
    parser.add_argument('--_gpu_assigned', action='store_true', help='gpu_assigned')
    
    parser.add_argument('--experiment_name', type=str, default='TEST')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--loss_type', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--clip_max_norm', type=float, default=None)
    
    parser.add_argument('--optimizer.type', type=str)
    parser.add_argument('--optimizer.learning_rate', type=float)
    parser.add_argument('--optimizer.lr_milestone_epoch', type=tuple)
    parser.add_argument('--optimizer.lr_gamma', type=float)
    
    parser.add_argument('--dataset.train_dataset', type=str)
    parser.add_argument('--dataset.valid_dataset', type=str)
    parser.add_argument('--dataset.train_batch_size', type=int)
    parser.add_argument('--dataset.valid_batch_size', type=int)
    parser.add_argument('--dataset.num_workers', type=int)
    
    parser.add_argument('--logging.save_dir', type=str)
    parser.add_argument('--logging.save_model', action='store_true')
    parser.add_argument('--logging.save_every', type=int)
    parser.add_argument('--logging.pretrain_ckpt', type=str, default=None)

    parser.add_argument('--model.type', type=str)
    parser = model_class.get_parser(parser)

    args = parser.parse_args(argv)
    return args
