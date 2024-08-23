import argparse
import json
import os

def load_config():    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-region", type=str, required=True, help="can be either 'brain' or 'abdom'.")
    parser.add_argument("-mode", type=str, default="pixel", help="can be either 'pixel' or 'sample'.", required=False)
    parser.add_argument("-workspace", type=str, required=True)

    args = parser.parse_args()
    
    # Load configuration from JSON file
    with open(os.path.join(args.workspace, 'submission_config.json'), 'r') as config_file:
        config = json.load(config_file)
    
    # Create an argparse.Namespace object from the configuration dictionary
    config = argparse.Namespace(**config)
    

    config.result_dir = args.output
    config.region = args.region
    config.mode = args.mode
    config.data_dir_val = args.input

    if config.region == "brain":
        config.mask_th =   config.mask_th_brain
    elif config.region == 'abdom':
        config.mask_th =  config.mask_th_abdom


    config.checkpoints_dir = os.path.join(args.workspace, config.checkpoints_dir)
    config.training_stats = os.path.join(args.workspace, config.training_stats)    
    config.hist_dir = os.path.join(args.workspace, config.hist_dir) 
    config.masks_dir = os.path.join(args.workspace, config.masks_dir)
    config.lpips_model = os.path.join(args.workspace, config.lpips_model)
    config.alex_weights = os.path.join(args.workspace, config.alex_weights)
    
    return config