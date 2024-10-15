import os
import argparse
import numpy as np

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.stop import STOP, MLP
from src.engines.stop_engine import STOPEngine
from src.utils.args import get_star_config
from src.utils.dataloader import load_dataset, get_dataset_info, load_adj_from_numpy
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def cont_learning(model, save_path, args):
    filename = 'final_model_s{}_{}_maxratio{}_c{}_core{}.pt'.format(args.seed, 
                                                                 args.year, 
                                                                 args.max_increase_ratio, 
                                                                 args.c,
                                                                 args.core)
    model.load_state_dict(torch.load(
        os.path.join(save_path, filename), map_location=args.device))
    return model


def get_config():
    parser = get_star_config()
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--model_dim', type=int, default=64)
    parser.add_argument('--prompt_dim', type=int, default=32)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--hid_dim', type=int, default=256)

    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--beta', type=float, default=1) # L = L_ob + gammaL_un'

    # tranining parameters
    parser.add_argument('--lrate', type=float, default=2e-3)
    parser.add_argument('--wdecay', type=float, default=1e-5)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    parser.add_argument('--extra_type', type=int, default=1)
    parser.add_argument('--same', type=int, default=0)
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record.log')
    logger.info(args)
    
    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    device = torch.device(args.device)
    data_path, adj_path, node_num = get_dataset_info(args.years)

    base = MLP(input_dim=args.input_dim,
                        output_dim=args.output_dim,
                        num_layer=args.num_layer, 
                        model_dim=args.model_dim, 
                        prompt_dim=args.prompt_dim, 
                        tod_size=288, 
                        kernel_size=args.kernel_size)
    model_dim = args.model_dim + 2*args.prompt_dim
    model = STOP(input_dim=args.input_dim,
                output_dim=args.output_dim,
                model_args=vars(args),
                stmodel=base,
                dim=[model_dim, model_dim],
                core=args.core,
                head=args.head)
    if args.ct:
        try:
            model = cont_learning(model, log_dir, args)
        except:
            print('No pretrained model!')




    if args.checkall:
        print('!!!!!!start check all!!!!!!!!')
        for year in ['2011', '2012', '2013', '2014', '2015', '2016', '2017']:
            args.checkyears = year
            _, adj_path_test, node_num_test = get_dataset_info(args.checkyears)
            ###adj###
            logger.info('Adj path: ' + adj_path)
            adj = load_adj_from_numpy(adj_path) # with self-loop
            # Get dataset information and node about training and test
            set_seed(args.seed) # Given random seed
            node_order = np.arange(node_num_test)
            node_training = node_order[:node_num]
            node_test = node_order
            node_num_test_difference = node_num_test - node_num
            node_frechet = node_order
            # spatial side information embedding
            
            dataloader, scaler = load_dataset(data_path, args, logger)

            
            
            loss_fn = masked_mae
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay, eps=1e-8)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                            milestones=[30, 80, 160, 250], 
                                                            gamma=0.5)

            s_mask = None
            optimizer_s = None
            scheduler_s = None

            engine = STOPEngine(device=device,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                s_mask=s_mask,
                                optimizer_s=optimizer_s,
                                scheduler_s=scheduler_s,
                                node_training=node_training,
                                node_test=node_test if args.sood else node_training,
                                node_sood_num=node_num_test_difference if args.sood else 0, 
                                node_frechet=node_frechet,
                                dataloader=dataloader,
                                scaler=scaler,
                                sampler=None,
                                loss_fn=loss_fn,
                                log_dir=log_dir,
                                logger=logger,
                                args=args,
                                )
            engine.evaluate('test')
    else:
        _, adj_path_test, node_num_test = get_dataset_info(args.checkyears)
        ###adj###
        logger.info('Adj path: ' + adj_path)
        adj = load_adj_from_numpy(adj_path) # with self-loop

        # Get dataset information and node about training and test
        set_seed(args.seed) # Given random seed
        node_order = np.arange(node_num_test)
        node_training = node_order[:node_num]
        node_test = node_order
        node_num_test_difference = node_num_test - node_num
        node_frechet = node_order
        
        dataloader, scaler = load_dataset(data_path, args, logger)
        
        loss_fn = masked_mae
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                        milestones=[30, 80, 160, 250], 
                                                        gamma=0.5)

        s_mask = None
        optimizer_s = None
        scheduler_s = None

        engine = STOPEngine(device=device,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            s_mask=s_mask,
                            optimizer_s=optimizer_s,
                            scheduler_s=scheduler_s,
                            node_training=node_training,
                            node_test=node_test if args.sood else node_training,
                            node_sood_num=node_num_test_difference if args.sood else 0, 
                            node_frechet=node_frechet,
                            dataloader=dataloader,
                            scaler=scaler,
                            sampler=None,
                            loss_fn=loss_fn,
                            log_dir=log_dir,
                            logger=logger,
                            args=args,
                            )


        if args.mode == 'train':
            engine.train()
        else:
            engine.evaluate(args.mode)


if __name__ == "__main__":
    main()