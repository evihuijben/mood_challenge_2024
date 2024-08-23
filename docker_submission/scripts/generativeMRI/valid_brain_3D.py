from src import logger
import pandas as pd
import nibabel as nib
import torchio as tio

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.utils import set_determinism
import torch

from pathlib import Path

from src.data.get_train_and_val_dataloader import get_data_loader
from postprocessing_utils import normalize_image, create_patient_mask, load_general_mask, CustomMetrics, calculate_merics_and_snapshot
from src.model import define_AE

from src.options import ValOptions

##############
args = ValOptions().parse()
saveto = os.path.join(args.results_folder, 
                    args.vae_name, 
                    f"epoch{args.vae_epoch}", 
                    Path(args.data_dir_val).name )


logger.configure(args, dir=saveto, format_strs=['log'])
##############


set_determinism(seed=args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed) # if use multi-GPU

logger.log("#"*50)
logger.log("loading data".center(50))
logger.log("#"*50 + "\n")
val_loader = get_data_loader(args.data_dir_val,args, shuffle=False )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.log(f"Using {device}")
                                  
vae_model = define_AE(args, device, discriminator=False)
checkpoint_path= os.path.join(args.output_dir, args.vae_name ,"checkpointG_{}.pth".format(args.vae_epoch))
checkpoint = torch.load(checkpoint_path, map_location=device)
vae_model.load_state_dict(checkpoint["model_state_dict"])
logger.log(f"Loaded VAE checkpoint: {checkpoint_path}")
vae_model.eval()




logger.log("Results being saved to", saveto)

general_mask = load_general_mask(args.general_mask)
logger.log('loaded general mask', args.general_mask)


os.makedirs(saveto, exist_ok=True)
recons_folder = os.path.join(saveto, 'recon')
latent_folder = os.path.join(saveto, 'latent')
snapshots_folder = os.path.join(saveto, f"snapshots__{'_'.join(args.metric_list)}")

os.makedirs(recons_folder, exist_ok=True)
os.makedirs(snapshots_folder, exist_ok=True)


#%%

def forward_pass(image, args, recon_fname):
    nonzero_slices = image.sum((0,1,2)).nonzero()[:,0]
    full_reconstruction = torch.zeros_like(image)
    for nz_slice in range(0, len(nonzero_slices), args.sub_batch_size):
        slices_this_batch = nonzero_slices[nz_slice : min( [nz_slice+args.sub_batch_size, len(nonzero_slices)]) ]
        image_nz_batch = image[...,slices_this_batch].permute(3,0,1,2).float().to(device)

        z_mu, _ = vae_model.encode(image_nz_batch)
        reconstruction_batch = vae_model.decode(z_mu)
        
        full_reconstruction[...,slices_this_batch] = reconstruction_batch.permute(1,2,3,0).detach().cpu()

    if args.save_recon == True:
        logger.log('\t\tsaving reconstruction')
        nifti_recon = nib.Nifti1Image(full_reconstruction.squeeze().numpy(), affine=np.eye(4))
        nib.save(nifti_recon, recon_fname)

    return full_reconstruction




my_metrics = CustomMetrics(args.metric_list, slice_axis=2)


all_results = {'2D': [], '3D': []}
logger.log('starting loop')
logger.log('saving reconstructions', args.save_recon)
with torch.no_grad():
    for val_step, batch in enumerate(val_loader):
        for image, path in zip(batch["image"], batch["path"]):
            fname = Path(path).name
            
            if image.shape != (1, args.isize, args.isize, args.isize, ):
                image = tio.Resize((args.isize, args.isize, args.isize, ))(image)

            if args.normalize == True:
                image = normalize_image(image)

            
            recon_fname = os.path.join(recons_folder, fname)
            if os.path.exists(recon_fname):
                logger.log(val_step, fname, 'reconstruction already exists')
                try:                    
                    full_reconstruction = nib.load(recon_fname).get_fdata()
                    full_reconstruction = torch.from_numpy(full_reconstruction).unsqueeze(0).float()
                except:
                    logger.log(val_step, fname, 'reconstruction corrupt, reprocessing ...')
                    full_reconstruction = forward_pass(image, args, recon_fname)
            else:
                logger.log(val_step, fname, 'processing ...')
                full_reconstruction = forward_pass(image, args, recon_fname)
                

            # final mask is union between patient_mask and general mask
            patient_mask = create_patient_mask(image, args.region, args.mask_th)
            mask = np.logical_or(general_mask, patient_mask)

            patient_results = calculate_merics_and_snapshot(
                fname,
                my_metrics,
                image,  
                full_reconstruction, 
                mask, 
                snapshots_folder,
                            )

            for dim in patient_results.keys():
                all_results[dim].extend(patient_results[dim])


for dim in all_results.keys():
    df = pd.DataFrame(all_results[dim])
    df.to_csv(os.path.join(saveto, f"Recon{dim}__{'_'.join(args.metric_list)}.csv"), index=False)


logger.log('finished')