import os
import argparse
import numpy as np

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.stop import STOP, MLP
from src.engines.stop_engine import Engine
from src.utils.args import get_star_config
from src.utils.dataset import load_dataset
from src.utils.graph import Graph
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger

def thresholded_Gaussian_kernel(adj, epsilon=0.01):
    std = np.std(adj[adj!=0])
    adj[adj!=0] = np.exp(-1 * (adj[adj!=0] / std)**2)
    adj[adj<epsilon] = 0
    np.fill_diagonal(adj, 1)
    return adj

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
    graph = Graph()
    adj = graph.attr_adj_dist
    adj = thresholded_Gaussian_kernel(adj)
    args.node_num = node_num = 184

    # Get dataset information and node about training and test
    set_seed(args.seed) # Given random seed
    node_order = np.arange(node_num)
    # np.save('/home/mjm/LST/LargeST-main/TKDE/LargeST/node_order.npy', node_order)
    # sys.exit()
    np.random.shuffle(node_order) # shuffle the node indices for ood segment
    node_num_training = int(node_num / (1.+args.max_increase_ratio))
    print('Using', node_num_training, 'nodes in training')
    node_num_test_increase = int(node_num_training * args.test_increase_ratio)
    node_num_test_decrease = int(node_num_training * args.test_decrease_ratio)
    # node_num_test_difference = node_num_test_increase + node_num_test_decrease
    node_num_test_difference = node_num_test_increase
    node_training = node_order[:node_num_training]
    node_test = np.concatenate([node_order[:node_num_training-node_num_test_decrease], 
                                node_order[node_num_training:node_num_training+node_num_test_increase]])
    node_frechet = node_order[:node_num_training+node_num_test_increase]
    # spatial side information embedding


    base = MLP(# node_num=node_num,
                input_dim=args.input_dim,
                output_dim=args.output_dim,
                num_layer=args.num_layer, 
                model_dim=args.model_dim, 
                prompt_dim=args.prompt_dim, 
                tod_size=8, 
                kernel_size=args.kernel_size)
    model_dim = args.model_dim
    model = STOP(# node_num=node_num,
                 input_dim=args.input_dim,
                 output_dim=args.output_dim,
                 model_args=vars(args),
                 stmodel=base,
                 dim=[model_dim, model_dim],
                 core=args.core,
                 ssie_dim=None,
                 head=args.head)
    if args.ct:
        try:
            model = cont_learning(model, log_dir, args)
        except:
            print('No pretrained model!')
    
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     milestones=[30, 80, 160], 
                                                     gamma=0.5)

    s_mask = None
    optimizer_s = None
    scheduler_s = None
    if args.checkall:
        print('!!!!!!!!!!start check all!!!!!!!!!!!!!')
        for year in range(2015, 2018+1):
            args.checkyears = year
            dataloader, scaler = load_dataset(graph, args.seq_len, args.horizon, args.checkyears, args.bs)

            engine = Engine(device=device,
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
                                # adj=adj,
                                ssie=None,
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
        dataloader, scaler = load_dataset(graph, args.seq_len, args.horizon, args.checkyears, args.bs)

        engine = Engine(device=device,
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
                            # adj=adj,
                            ssie=None,
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