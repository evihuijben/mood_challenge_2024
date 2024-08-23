from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import torch

import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append('/workspace/generativeMRI')
sys.path.append('docker_submission/scripts')
sys.path.append('docker_submission/scripts/generativeMRI')
sys.path.append('generativeMRI')


import matplotlib.pyplot as plt
import pandas as pd

from generativeMRI.postprocessing_utils import normalize_image, create_patient_mask, load_general_mask
from generativeMRI.src.model import AutoEncoderNoKL, AutoencoderKL

import lpips
from lpips_for_docker import LPIPS_readonly
from data import upsample




class CustomMetricsMOOD2024():
    def __init__(self, args, metric_list, device, slice_axis=0):
        self.args = args
        self.metric_list = metric_list
        if 'LPIPS' in self.metric_list:
            # READ only LPIPS is created to avoid downloading the model weights in read only mode
            # without docker, the model weights can be downloaded and used directly using loss_fn = lpips.LPIPS(spatial=True)
            if os.path.isfile(args.lpips_model) and os.path.isfile(args.alex_weights):
                loss_fn = LPIPS_readonly(alex_weights=args.alex_weights ,model_path=args.lpips_model,spatial=True)
            else:
                loss_fn = lpips.LPIPS(spatial=True)
            if torch.cuda.is_available():
                loss_fn = loss_fn.cuda()
            self.loss_fn = loss_fn

        self.slice_axis = slice_axis
        if slice_axis == 0:
            self.agg_axis = (1,2)
        elif slice_axis == 2:
            self.agg_axis = (0,1)
        else:
            raise NotImplementedError('Slice axis 1 not implemented') 
        
        training_stats = pd.read_csv(args.training_stats)
        training_stats = training_stats.sort_values(by='Slice_')
        self.training_stats = {model_name: training_stats[training_stats['model_'] == model_name] 
                               for model_name in training_stats['model_'].unique()}


    def MAE(self, input, recon):
        abs_diff =  torch.abs(input - recon)
        mae = abs_diff.mean().item()
        return abs_diff


    def LPIPS(self, input, recon):
        batchsize = self.args.sub_batch_size

        nonzero_slices = input.sum(self.agg_axis).nonzero()[:,0]
        input_img = input[...,nonzero_slices]
        recon_img = recon[...,nonzero_slices]

        input_img = (input_img*255).type(torch.uint8).unsqueeze(0)
        recon_img = (recon_img*255).type(torch.uint8).unsqueeze(0)
        
        if self.slice_axis == 2:
            input_img = input_img.permute(3,0,1,2)
            recon_img = recon_img.permute(3,0,1,2)
        elif self.slice_axis == 0:
            input_img = input_img.permute(1,0,2,3)
            recon_img = recon_img.permute(1,0,2,3)

        
        input_img = input_img.repeat(1,3,1,1)
        recon_img = recon_img.repeat(1,3,1,1)


        
        distances = []
        for pos in range(0, input_img.shape[0], batchsize):
            start = pos 
            end = min(start + batchsize , input_img.shape[0])
            d =  self.loss_fn.forward(input_img[start:end], recon_img[start:end])

            distances.append(d)

        distances = torch.cat(distances)
        if self.slice_axis == 2:
            distances = distances.permute(1,2,3,0)[0]
        elif self.slice_axis == 0:
            distances = distances[:,0]
            
        # return distances
        distances_map_full = torch.zeros_like(input)
        distances_map_full[...,nonzero_slices] = distances
        return distances_map_full
    
    def standardize(self, input, model_name, metric, epsilon=1e-12):
        output = np.zeros_like(input)
        for slice in range(input.shape[2]):


            this_mean = self.training_stats[model_name][f'{metric}_mean'].values
            this_std = self.training_stats[model_name][f'{metric}_std'].values
    
            # Reshape the means and stds to match the input dimensions
            this_mean = this_mean.reshape((1, 1, input.shape[2]))
            this_std = this_std.reshape((1, 1, input.shape[2]))
            
            # Standardize the input, adding epsilon to the std to avoid division by zero
            output = (input - this_mean) / (this_std + epsilon)
            
        return output

class ReconstructMoodSubmission():
    def __init__(self, args):
        self.args = args


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device defined, using', self.device)

        general_mask_path = os.path.join(args.masks_dir, f"{args.region}_train_mask.nii.gz")
        self.general_mask = load_general_mask(general_mask_path)
        print('Loaded general mask from', general_mask_path )

        self.models = self.define_networks()
        print('Initialized networks')

        self.my_metrics = CustomMetricsMOOD2024(args, args.metric_list, self.device, slice_axis=2)
        print('Initialized metrics')


        self.easy_detection = args.easy_detection
        if self.easy_detection:
            self.n_bins = 4096
            self.all_hist = np.load(os.path.join(args.hist_dir, f'all_hist_{args.region}{self.n_bins}.npy'))
            assert self.all_hist.shape==self.all_hist.shape

    def define_networks(self):

        if self.args.kl_weight == 0:
            ae = AutoEncoderNoKL
        else:
            ae = AutoencoderKL


        vae_model2x2 = ae(spatial_dims = 2,
                          in_channels = 1 if self.args.is_grayscale else 3,
                          out_channels = 1 if self.args.is_grayscale else 3,
                          num_channels = [128, 128, 256, 256, 512, 512, 512, 512],
                          latent_channels = 32,
                          num_res_blocks = 2,
                          norm_num_groups = 32,
                          attention_levels = [False, False, False, False, False, False, False, True],
                          )
        vae_model2x2.to(self.device)
        checkpoint_path2x2= os.path.join(self.args.checkpoints_dir, f"{self.args.region}_32x2x2" ,"checkpointG_{}.pth".format(self.args.vae_epoch))
        checkpoint2x2 = torch.load(checkpoint_path2x2, map_location=self.device)
        vae_model2x2.load_state_dict(checkpoint2x2["model_state_dict"])
        vae_model2x2.eval()
        print(f"Initialized model 2x2. Loaded VAE checkpoint: {checkpoint_path2x2}")



        vae_model8x8 = ae(spatial_dims = 2,
                          in_channels = 1 if self.args.is_grayscale else 3,
                          out_channels = 1 if self.args.is_grayscale else 3,
                          num_channels = [128, 128, 256, 256, 512, 512],
                          latent_channels = 32,
                          num_res_blocks = 2,
                          norm_num_groups = 32,
                          attention_levels = [False, False, False, False, False, True],
                          )
        vae_model8x8.to(self.device)
        checkpoint_path8x8 = os.path.join(self.args.checkpoints_dir, f"{self.args.region}_32x8x8" ,"checkpointG_{}.pth".format(self.args.vae_epoch))
        checkpoint8x8 = torch.load(checkpoint_path8x8, map_location=self.device)
        vae_model8x8.load_state_dict(checkpoint8x8["model_state_dict"])
        vae_model8x8.eval()
        print(f"Initialized model 8x8. Loaded VAE checkpoint: {checkpoint_path8x8}")

        return {'32x2x2': vae_model2x2, '32x8x8': vae_model8x8}


    def forward_pass(self, model, image):
        sub_batch_size = self.args.sub_batch_size
        nonzero_slices = image.sum((0,1,2)).nonzero()[:,0]
        full_reconstruction = torch.zeros_like(image)
        for nz_slice in range(0, len(nonzero_slices), sub_batch_size):
            slices_this_batch = nonzero_slices[nz_slice : min( [nz_slice+sub_batch_size, len(nonzero_slices)]) ]
            image_nz_batch = image[...,slices_this_batch].permute(3,0,1,2).float()

            z_mu, _ = model.encode(image_nz_batch)
            reconstruction_batch = model.decode(z_mu)

            full_reconstruction[...,slices_this_batch] = reconstruction_batch.permute(1,2,3,0).detach()

        return full_reconstruction

    def vae_localization(self, image):
        image = image.to(self.device)
        if self.args.normalize == True:
            image = normalize_image(image)

        recons = {}
        for key, model in self.models.items():
            recons[key] = self.forward_pass(model, image)


        patient_mask = create_patient_mask(image, self.args.region, self.args.mask_th)
        mask = np.logical_or(self.general_mask, patient_mask)


        result = self.process_reconstruction(image, recons, mask)
        return result





    def process_reconstruction(self, input, recons, mask):
        input = input.squeeze().clip(0,1)


        all_results = []
        print('    Calculating metrics ...')

        for key, recon in recons.items():
            recon = recon.squeeze().clip(0,1)


            for metric in self.my_metrics.metric_list:
                result_map = getattr(self.my_metrics, metric)(input, recon)
                result_map_standardized = self.my_metrics.standardize(result_map.cpu().numpy(), f"{self.args.region}_{key}", metric)

                if metric == 'MAE':
                  result_map_standardized=gaussian_filter(result_map_standardized, 3)


                print(f'\t\t{key} {metric} = {result_map.mean().item()}')

                all_results.append(result_map_standardized)

        combined = np.stack(all_results, axis=0).mean(axis=0).astype(np.float32)
        
        # Normalize the combined map
        minval, maxval = 20, 50
        combined_norm = np.clip(combined, minval, maxval) 
        combined_norm = (combined_norm - minval) / (maxval - minval)

        combined_norm_mask = np.where(mask, combined_norm, 0.0)
        return combined_norm_mask



        
    def easy_localization(self, batch, n_std_outlier=4):
        image = batch["image"].float()#[tio.DATA].float()
        if self.args.region == 'brain':
            std_th = 64
        else:
            std_th = 128
        
        # Calculate the mean and std of all training histograms and adjust the std to a predefined threshold
        hist_mean = np.mean(self.all_hist, axis=0, keepdims=True).squeeze()
        hist_std = np.std(self.all_hist, axis=0, keepdims=True).squeeze()
        hist_std[hist_std<std_th] = std_th
        
        
        
        # Prepare image data and calculate histogram for this case
        image_data = image.cpu().numpy().squeeze()
        image_flatened = image_data.flatten()
        hist, bins = np.histogram(image_flatened, bins=self.n_bins, range=(0, 1))


        # Check for outliers intensity values
        sub_hist = np.abs(hist - hist_mean) 
        sub_hist[sub_hist <= n_std_outlier * hist_std] = 0 
        spike_inds = sub_hist.nonzero()[0]
        spike_inds.sort()


        # Remove spike at zero (background)
        if len(spike_inds) > 0  and spike_inds[0] == 0:
            spike_inds = spike_inds[1:]

        # Threshold image based on outlier spikes
        thresholded_image = np.zeros(image_data.shape, dtype=np.uint8)
        if len(spike_inds)> 0:
            
            # Find connected outlier bins to speed up the process
            connected_groups = []
            current_group = [spike_inds[0]]
            for i in range(1,len(spike_inds)):
                if spike_inds[i] == spike_inds[i-1] +1:
                    current_group.append(spike_inds[i])
                else:
                    connected_groups.append(current_group)
                    current_group = [spike_inds[i]]
            connected_groups.append(current_group)
            
            # Threshold image based on this group's lower and upper threshold
            for group in connected_groups:
                first_ind = group[0]
                last_ind = group[-1]
                
                threshold_lower = bins[first_ind]
                threshold_upper = bins[last_ind+1]
                    
                thresholded_image += ((image_data > threshold_lower) & (image_data < threshold_upper)).astype(np.uint8)
            
            # Apply erosion and dilation on thesholded image
            structure_size = 6
            thresholded_image = binary_erosion(thresholded_image, np.ones((structure_size,structure_size,structure_size)))
            thresholded_image = binary_dilation(thresholded_image, np.ones((structure_size,structure_size,structure_size)))
            
            return thresholded_image.astype(np.uint8)
        else:
            return thresholded_image

    def score_one_case(self, batch):
        print(batch["image"].shape)
        if self.easy_detection:
            ############################
            # easy detection
            ############################
            print('    Easy local started ...', batch['path'])
            easy_result = self.easy_localization(batch)
            print('    Easy local finished. sum=', easy_result.sum() )
             
            
            if easy_result.any():
                print('    Easy local found!')
                if self.args.region == 'brain':
                    print('returning easy brain result')
                    return easy_result, 'easy'
                else:
                    print('returning easy abdom result')
                    return upsample(easy_result, interpolation='nearest'), 'easy'

        ############################
        # vae detection
        ############################
        vae_result = self.vae_localization(batch["image"][0])
         
        
        if self.args.region == 'brain':
            print('returning vae brain result')
            return vae_result , 'vae'
        else:
            print('returning vae abdom result')
            return upsample(vae_result), 'vae'