print('python started')
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.utils import set_determinism
import torchio as tio
import wandb
from pathlib import Path


from src.options import TrainOptions
from src.model import define_AE, myLosses, save_checkpoints
from src.data.get_train_and_val_dataloader import get_data_loader


args = TrainOptions().parse()
set_determinism(seed=args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed) # if use multi-GPU

### Load data
print("#"*50)
print("loading data".center(50))
print("#"*50)
train_loader = get_data_loader(args.data_dir, args, shuffle=True, )
val_loader = get_data_loader(args.data_dir_val,args, shuffle=False )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

model, discriminator = define_AE(args, device)
if args.ldm_name !='':
    raise NotImplementedError("LDM not implemented, see train_ldm_brain.py")
loss = myLosses(args, device)
optimizer_g = torch.optim.Adam(params=model.parameters(), lr=args.lr_G)
optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=args.lr_D)


model_id = f"{time.strftime('%y%m%d_%H%M%S')}_{args.vae_name}"

run_dir = Path(args.output_dir) / model_id
os.makedirs(run_dir, exist_ok=True)


wandb.init(project=args.project, name=model_id, config=args, entity=args.wandb_entity)



print("#"*50)
print("model training".center(50))
print("#"*50)


def make_plot(images, reconstruction):
    
    to_plot_input = images[0].squeeze().detach().cpu().numpy()
    to_plot_recon = reconstruction[0].squeeze().detach().cpu().numpy()

    fig, ax = plt.subplots(1,2, figsize=(12,10))

    ax[0].imshow(to_plot_input, cmap='gray', vmin=0, vmax=1)
    ax[0].axis('off')
    ax[0].set_title('Input')

    ax[1].imshow(to_plot_recon, cmap='gray', vmin=0, vmax=1)
    ax[1].axis('off')
    ax[1].set_title("Reconstruction")
    return fig


total_start = time.time()
best_loss = 100000
for epoch in range(args.n_epochs):
    model.train()
    discriminator.train()

    print("Training step started epoch ", epoch)
    for step, batch in enumerate(train_loader):
        if step % 5000 == 0:
            print(f"Step {step}/{len(train_loader)}")
        
        images = batch["image"].float().to(device)
        optimizer_g.zero_grad(set_to_none=True)

        reconstruction, z_mu, z_sigma = model(images)
        
        # reconstruction = F.tanh(reconstruction)
        logits_fake = discriminator(reconstruction.contiguous().detach())[-1]

        losses_dict = loss.calculate_generator_losses(reconstruction, images, z_mu, z_sigma, logits_fake)
        losses_dict["gen_loss_all"].backward()
        optimizer_g.step()

        # Discriminator part
        optimizer_d.zero_grad(set_to_none=True)

        logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
        logits_real = discriminator(images.contiguous().detach())[-1]

        disc_losses = loss.calculate_discriminator_losses(logits_fake, logits_real)
        losses_dict.update(disc_losses)
        losses_dict["discr_loss"].backward()
        optimizer_d.step()

        wandb.log({f'train/{k}':v.item() for k, v in losses_dict.items()})
        #### log to wandb
        if step%100== 0:
            fig = make_plot(images, reconstruction)
            wandb.log({"training_recons": fig})
            plt.close()

    print(f"Saving model for epoch {epoch}")      
    save_checkpoints(model, discriminator, run_dir, epoch, suffix="last")

    print("Validation step starts")
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_step, batch in enumerate(val_loader):
            images = batch["image"].float().to(device)
            reconstruction, z_mu, z_sigma = model(images)
            # reconstruction = F.tanh(reconstruction)            
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            logits_real = discriminator(images.contiguous().float())[-1]
            val_losses_dict = loss.calculate_generator_losses(reconstruction, images, z_mu, z_sigma, logits_fake)
            val_losses_dict.update(loss.calculate_discriminator_losses(logits_fake, logits_real))


            val_loss += val_losses_dict["gen_loss_all"].item()
            wandb.log({f'val/{k}':v.item() for k, v in val_losses_dict.items()})
            if val_step%100== 0:
                fig = make_plot(images, reconstruction)
                wandb.log({"validation_recons": fig})
                plt.close()

    val_loss /= (val_step+1)
    print("Validation step ends")

    # save the best model
    if val_loss < best_loss:
        best_loss = val_loss
        print(f"Best epoch. Saving model with val loss {val_loss}")
        save_checkpoints(model, discriminator, run_dir, epoch, suffix="best")

total_time = time.time() - total_start
wandb.finish()
print(f"train completed, total time: {total_time}.")
