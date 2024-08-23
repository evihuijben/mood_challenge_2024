import os
import tempfile
custom_tempdir = '/mnt/pred/tmp'
tempfile.tempdir = custom_tempdir
print()
print()
print('Using tmp dir', custom_tempdir)
if os.path.isdir('/mnt/pred'):
    print('listdir /mnt/pred', os.listdir('/mnt/pred'))
    for sub in os.listdir('/mnt/pred'):
        if os.path.isdir(sub):
            print(sub , 'ls:', os.listdir(os.path.join('/mnt/pred', sub)))
else:
    print('/mnt/pred not existing')
    

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append('/workspace')
sys.path.append('docker_submission/scripts')


import nibabel as nib
from pathlib import Path
import time
import torch
from monai.utils import set_determinism
import numpy as np


from options import load_config
from mood_model import ReconstructMoodSubmission
from data import get_data_loader
# from generativeMRI.src.data.get_train_and_val_dataloader import get_data_loader


def predict(args, my_model, loader):
    
    print('Processing cases ...')
    with torch.no_grad():
        for val_step, batch in enumerate(loader):
            print()
            t_start = time.time()
            if args.batch_size == 1:
                pred_local, which_model = my_model.score_one_case(batch)

                f = Path(batch['path'][0]).name

                if args.mode == "pixel":
                    target_file = os.path.join(args.result_dir, f)
                    # affine = batch['image']['affine'][0]  # idx 0 in batch
                    affine = np.eye(4)
                    final_nib_img = nib.Nifti1Image(pred_local, affine=affine)
                    nib.save(final_nib_img, target_file)
                    
                elif args.mode == "sample":
                    if which_model == 'easy':
                        if pred_local.any():
                            abnomal_score = 1.0
                        else:
                            abnomal_score = 0.0
                    elif which_model == 'vae':
                        th = 0
                        thresholded = np.where(pred_local> th, 1.0, 0.0 )
                        percentage = 100 * thresholded.sum() / thresholded.size

                        p = 1.0 if args.region == 'brain' else 0.1

                        if percentage > p:
                            abnomal_score = 1.0
                        else:
                            abnomal_score = 0.0

                        
                    with open(os.path.join(args.result_dir, f + ".txt"), "w") as write_file:
                        write_file.write(str(abnomal_score))   
                    print('Sample saved to', os.path.join(args.result_dir, f + ".txt"), os.listdir(args.result_dir))  
                else:
                    print("Mode not correctly defined. Either choose 'pixel' oder 'sample'")
            else:
                raise NotImplementedError()
            print(f'Case {val_step} ({f}) took {time.time()-t_start:.2f} seconds. Sum of prediction = {pred_local.sum()}')
    print('Finished')

        
        
if __name__ == "__main__":

    print('>>> Loading config ...')
    args = load_config()
    set_determinism(seed=args.seed)
    print('>>> Config loaded')
    print()

    print('>>> Init model ...')
    my_model = ReconstructMoodSubmission(args)
    print('>>> Model initialized')
    print()

    print('>>> Loading data ...' )
    loader = get_data_loader(args.data_dir_val, args, shuffle=False )
    print('>>> Dataloader created')
    print()

    
    print('>>> Start predict ...')
    predict(args, my_model, loader)
    print('>>> Finished predict')
    