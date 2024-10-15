import argparse

def get_star_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--dataset', type=str, default='traffic_default')
    # if need to use the data from multiple years, please use underline to separate them, e.g., 2018_2019
    parser.add_argument('--years', type=str, default='2011')
    parser.add_argument('--checkyears', type=str, default='2012')
    parser.add_argument('--model_name', type=str, default='stop')
    parser.add_argument('--seed', type=int, default=3028)
    parser.add_argument('--bs', type=int, default=64)

    # for basic training
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--output_dim', type=int, default=1)

    # for data segment
    parser.add_argument('--max_increase_ratio', type=float, default=0.3) # dataset-driven hyperpara.
    parser.add_argument('--checkall', type=int, default=0) # speed up
    
    # for traning
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--test_increase_ratio', type=float, default=0.1) # must less than max_changable_ratio
    parser.add_argument('--test_decrease_ratio', type=float, default=0.1) # must less than 1
    parser.add_argument('--sood', type=int, default=1) # if spatial shift or not
    parser.add_argument('--tood', type=int, default=1) # if temporal shift or not
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=50)
    
    # For Frechet embbeding
    parser.add_argument('--c', type=int, default=1)
    parser.add_argument('--adj_type', type=str, default='origin')
    parser.add_argument('--spatial_noise', type=bool, default=False)

    # For GPU
    parser.add_argument('--mask_patience', type=int, default=5)
    parser.add_argument('--K_t', type=int, default=3) # num of t_mask
    parser.add_argument('--K_s', type=int, default=3) # num of t_mask
    parser.add_argument('--t_sample_ratio', type=float, default=0.1) # ratio of mask
    parser.add_argument('--s_sample_ratio', type=float, default=0.2) # ratio of mask

    # For CPU
    parser.add_argument('--core', type=int, default=8)
    parser.add_argument('--head', type=int, default=8)

    # For CPU
    parser.add_argument('--ct', type=int, default=0)
    return parser


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')
