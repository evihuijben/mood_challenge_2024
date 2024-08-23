import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchio as tio
import skimage
from scipy import ndimage
from skimage.metrics import structural_similarity
import nibabel as nib
from tqdm import tqdm
import pandas as pd
import lpips
import seaborn as sns
from pathlib import Path


def plot_overview(savefolder, fname, plot_inputs, all_maps, all_labels=None, sl=128, slice_axis =0 ):



    sns.set_theme()
    fig, ax = plt.subplots(1,len(plot_inputs)+len(all_maps), figsize=(len(plot_inputs)+len(all_maps)*3, 3))
    for i, (label, img) in enumerate(plot_inputs.items()):

        if slice_axis == 0:
            img = img[sl,:,:]
        elif slice_axis == 2:
            img = img[:,:,sl]

        obj = ax[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.colorbar(obj, ax=ax[i],fraction=0.046)
        ax[i].axis('off')
        ax[i].set_title(label)


    for i, (label, img) in enumerate(all_maps.items()):
        if 'ssim' in label.lower():
            cmap='RdYlGn'
            vmin, vmax = 0, 1
        elif 'mae' in label.lower():
            cmap='seismic'
            vmin, vmax = -1, 1
        elif 'lpips' in label.lower():
            cmap=None
            vmin, vmax = 0, 1

        if slice_axis == 0:
            img = img[sl,:,:]
        elif slice_axis == 2:
            img = img[:,:,sl]
        obj = ax[i+2].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(obj, ax=ax[i+2],fraction=0.046)
        ax[i+2].axis('off')
        
        if all_labels is not None:
            ax[i+2].set_title(all_labels[label])
        else:
            ax[i+2].set_title(label)
    plt.suptitle(f"{fname}")


    plt.tight_layout()
    plt.savefig(os.path.join(savefolder, f"{fname}.png"), bbox_inches='tight')
    #plt.show()
    plt.close()


def normalize_image(image):
    min_val = image.min()
    max_val = image.max()
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image

def load_general_mask(path_to_general_mask, threshold= 0.01):
    general_mask = tio.Subject(image=tio.ScalarImage(path_to_general_mask))['image'].data
    general_mask = general_mask.squeeze()
    general_mask = torch.where(general_mask > threshold, 1, 0)
    return general_mask.numpy()


def create_patient_mask(image, region, threshold):
    image = image.squeeze()
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()


    patient_mask = image>threshold

    patient_mask = ndimage.binary_erosion(patient_mask)
    patient_mask = ndimage.binary_dilation( patient_mask)           
    
    # Find the largest connected component
    labeled, ncomponents = ndimage.label(patient_mask)
    component_sizes = np.bincount(labeled.ravel())
    # Ignore background
    component_sizes[0] = 0
    patient_mask = (labeled == np.argmax(component_sizes)).astype(np.uint8)

    if region == 'abdom':
        print('filling holes')
        patient_mask = ndimage.binary_fill_holes(patient_mask)

    return patient_mask


class CustomMetrics():
    def __init__(self, metric_list, slice_axis=0, metric_kwargs=None):
        self.metric_list = metric_list
        if 'LPIPS' in self.metric_list:
            loss_fn = lpips.LPIPS(spatial=True)
            if torch.cuda.is_available():
                loss_fn = loss_fn.cuda()
            self.loss_fn = loss_fn

        if metric_kwargs is None:
            self.kwargs = {metric: {} for metric in self.metric_list}
        else:
            self.kwargs = metric_kwargs

        self.slice_axis = slice_axis
        if slice_axis == 0:
            self.agg_axis = (1,2)
        elif slice_axis == 2:
            self.agg_axis = (0,1)
        else:
            raise NotImplementedError('Slice axis 1 not implemented') 
        
        
    def MAE(self, input, recon, mask=None, clip=True):
        if clip:
            input = input.clip(0,1)
            recon = recon.clip(0,1)

        if isinstance(input, torch.Tensor):
            input = input.cpu().numpy()
        if isinstance(recon, torch.Tensor):
            recon = recon.cpu().numpy()

        difference = input - recon


        mae = np.mean(np.abs(difference))
        results = {'values_3D': {'MAE': mae}, 
                   'values_2D': {'MAE': np.nanmean(np.abs(difference), axis=self.agg_axis)},
                   'maps': {'MAE': difference}, 
                   'labels': {'MAE': f'Input-Recon (MAE={mae:.3f})'},
                   }
        if mask is not None:
            masked_difference = np.where(mask==1, difference, np.nan)
            masked_mae = np.nanmean(np.abs(masked_difference))
            results['values_3D']['MAE_masked'] = masked_mae
            results['values_2D']['MAE_masked'] = np.nanmean(np.abs(masked_difference), axis=self.agg_axis)
            results['maps']['MAE_masked'] = masked_difference
            results['labels']['MAE_masked'] = f'Masked Input-Recon (MAE={masked_mae:.3f})'
        return results


    def SSIM(self, input, recon, mask=None, clip=True, data_range=1):
        if clip:
            input = input.clip(0,1)
            recon = recon.clip(0,1)

        if isinstance(input, torch.Tensor):
            input = input.cpu().numpy()
        if isinstance(recon, torch.Tensor):
            recon = recon.cpu().numpy()

        mssim, ssim_map = structural_similarity(
            input,
            recon,
            data_range=data_range, 
            full=True
            )
            
        results = {'values_3D': {'SSIM': mssim},
                   'values_2D' : {'SSIM': np.nanmean(ssim_map, axis=self.agg_axis)},
                   'maps': {'SSIM': ssim_map},
                   'labels': {'SSIM': f'SSIM ({mssim:.2f})'},
                   }
        if mask is not None:
            masked_ssim_map = np.where(mask==1, ssim_map, np.nan)
            masked_ssim = np.nanmean(ssim_map)

            results['values_3D']['SSIM_masked'] = masked_ssim
            results['values_2D']['SSIM_masked'] = np.nanmean(masked_ssim_map, axis=self.agg_axis)
            results['maps']['SSIM_masked'] = masked_ssim_map
            results['labels']['SSIM_masked'] = f'Masked SSIM ({masked_ssim:.2f})'
        return results
        

    def LPIPS(self, input, recon, mask=None, batchsize=4, clip=True):
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)
            recon = torch.from_numpy(recon)
        if clip:
            input = input.clip(0,1)
            recon = recon.clip(0,1)

                
        input = (input*255).type(torch.uint8).unsqueeze(0)
        recon = (recon*255).type(torch.uint8).unsqueeze(0)
        
        if self.slice_axis == 2:
            input = input.permute(3,0,1,2)
            recon = recon.permute(3,0,1,2)
        elif self.slice_axis == 0:
            input = input.permute(1,0,2,3)
            recon = recon.permute(1,0,2,3)

        
        input_img = input.repeat(1,3,1,1)
        recon_img = recon.repeat(1,3,1,1)
        
        if torch.cuda.is_available():
            input_img = input_img.cuda()
            recon_img = recon_img.cuda()



        distances = []
        for pos in range(0, input_img.shape[0], batchsize):
            start = pos 
            end = min(start + batchsize , input_img.shape[0])
            d =  self.loss_fn.forward(input_img[start:end], recon_img[start:end]).cpu().detach()

            distances.append(d)

        distances = torch.cat(distances)
        if self.slice_axis == 2:
            distances = distances.permute(1,2,3,0)[0]
        elif self.slice_axis == 0:
            distances = distances[:,0]
            
        distances_map = distances.numpy()
        distances_mean = distances.mean().item()

        results = {
            'values_3D': {'LPIPS': distances_mean},
            'values_2D': {'LPIPS': np.nanmean(distances_map, axis=self.agg_axis )},
            'maps': {'LPIPS': distances_map},
            'labels': {'LPIPS': f'LPIPS ({distances_mean:.3f})'},
        }
        if mask is not None:
            masked_distances_map = np.where(mask==1, distances_map, np.nan)
            masked_distances_mean = np.nanmean(distances_map)

            results['values_3D']['LPIPS_masked'] = masked_distances_mean
            results['values_2D']['LPIPS_masked'] = np.nanmean(masked_distances_map, axis=self.agg_axis )
            results['maps']['LPIPS_masked'] = masked_distances_map
            results['labels']['LPIPS_masked'] = f'Masked LPIPS ({masked_distances_mean:.3f})'

        return results

def calculate_merics_and_snapshot(fname, my_metrics, input, recon, mask=None, snapshots_folder=None):
    input = input.squeeze()
    recon = recon.squeeze()

    
    all_results_3D = {'Filename': fname}
    all_results_2D = [{'Filename': fname, 'Slice': slice_i} for slice_i in range(input.shape[0])]

    all_maps = {}
    all_labels = {}

    for metric in my_metrics.metric_list:
        results = getattr(my_metrics, metric)(input, recon, mask, **my_metrics.kwargs[metric])  # keys: values, maps, labels


        all_results_3D.update(results['values_3D'])
        all_maps.update(results['maps'])
        all_labels.update(results['labels'])

        for slice_i in range(results['values_2D'][metric].shape[0]):
            for metric_with_suffix in results['values_2D'].keys():
                all_results_2D[slice_i][metric_with_suffix] = results['values_2D'][metric_with_suffix][slice_i]
    
    all_results_3D = [all_results_3D]

    if snapshots_folder is not None:
        plot_inputs = {'Image': input, 'Reconstruction': recon}
        os.makedirs(snapshots_folder, exist_ok=True)
        plot_overview(snapshots_folder,fname, plot_inputs, all_maps, all_labels, slice_axis=my_metrics.slice_axis)


    return {'3D': all_results_3D, '2D': all_results_2D} 

