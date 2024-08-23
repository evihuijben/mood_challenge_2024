import argparse


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Base Options")
        self.initialized = False

    def initialize(self):
        self.parser.add_argument("--data_dir_val", type=str, required=True, help="validation data directory.")
        self.parser.add_argument("--output_dir",  default='.checkpoints', help="Location for model checkpoints")
        self.parser.add_argument("--seed", default=42, help="seed for reproducibility")
        

        # Dataloader parameters
        self.parser.add_argument("--batch_size", type=int, default=24, help="Training batch size.")
        self.parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
        self.parser.add_argument("--is_grayscale", type=int, default=1, help="Is data grayscale.")

        # General model parameters
        self.parser.add_argument("--latent_channels", type=int, default=32, help="Number of latent channels.")
        self.parser.add_argument("--spatial_dimension", type=int, default=2, help="spatial dimension of images (2 for 2D, 3 for 3D).")
        self.parser.add_argument("--vae_name", default='VAE' , help="Name of vae model for encoding and decoding.")
        self.parser.add_argument("--vae_norm_num_groups", type=int, default=32,)
        self.parser.add_argument("--vae_num_res_blocks", type=int, default=2,)
        self.parser.add_argument("--vae_attention_levels", type=str, default="0,0,0,0,0,1",)
        self.parser.add_argument("--vae_num_chanels", type=str, default="128,128,256,256,512,512")
        self.parser.add_argument('--adv_weight', type=float, default=0.005, )
        self.parser.add_argument('--perceptual_weight', type=float, default=0.002,)
        self.parser.add_argument('--kl_weight', type=float, default=0)

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        self.opt.vae_attention_levels = [bool(int(x)) for x in self.opt.vae_attention_levels.split(',')]
        self.opt.vae_num_chanels = [int(x) for x in self.opt.vae_num_chanels.split(',')]

        return self.opt


class TrainOptions(BaseOptions):
    def initialize(self):
        super().initialize()
        self.parser.add_argument("--data_dir", type=str, required=True, help="training data directory.")
        self.parser.add_argument("--lr_G", type=float, default=1e-4, help="Learning rate for generator.")
        self.parser.add_argument("--lr_D", type=float, default=5e-4, help="Learning rate for discriminator.")

        self.parser.add_argument('--n_epochs', type=int, default=100)

        # Wandb parameters
        self.parser.add_argument("--project", type=str, required=True, help="wandb project name")
        self.parser.add_argument("--wandb_entity", type=str, required=True, help="wandb entity")
        
        self.initialized = True

class ValOptions(BaseOptions):
    def initialize(self):
        super().initialize()
        # Data parameters
        self.parser.add_argument("--results_folder", type=str, required=True, help="results folder.")
        self.parser.add_argument("--general_mask", type=str, required=True, help="Nifti file of general mask for the data.")
        self.parser.add_argument("--metric_list", default=['MAE','SSIM','LPIPS'])

        self.parser.add_argument("--normalize", action='store_true', help="normalize the data")
        self.parser.add_argument("--sub_batch_size", type=int, default=64, help="sub batch size for 3D inference")
        self.parser.add_argument("--vae_epoch", type=str, default="best" , help="epoch of the vae model for loading.")
        self.parser.add_argument("--isize", type=int, default=256)
        
        self.parser.add_argument("--region", type=str, default='brain')
        self.parser.add_argument("--mask_th", type=float, default=0.01)

        self.parser.add_argument("--save_recon", action='store_true', help="save the reconstructions")

        self.initialized = True


def print_args(args):
    print("*"*50)
    print("Arguments".center(50))
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print("*"*50)

if __name__ == '__main__':
    # Example of how to use TrainOptions and ValOptions

    # To parse training options
    train_args = TrainOptions().parse()
    print(train_args)

    # To parse validation options
    val_args = ValOptions().parse()
    print(val_args)


