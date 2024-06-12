import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset.shapenet_dataset import ShapeNetCore  
from models.encoders.pointnet import PointNetEncoder
from scheduler.linear_noise_scheduler import LinearNoiseScheduler, DiffusionPoint, PointwiseNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )

    # Create the dataset
    shapenet = ShapeNetCore(dataset_config['data_path'], dataset_config['categories'], dataset_config['split'],
                            dataset_config['scale_mode'])
    shapenet_loader = DataLoader(shapenet, batch_size=train_config['batch_size'], shuffle=True, num_workers=4)

    # Instantiate the models
    encoder = PointNetEncoder(zdim=model_config['latent_dim']).to(device)
    pointwise_net = PointwiseNet(point_dim=3, context_dim=model_config['latent_dim'], residual=model_config['residual']).to(device)
    diffusion = DiffusionPoint(net=pointwise_net, scheduler=scheduler).to(device)

    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    # Load checkpoint if found
    if os.path.exists(os.path.join(train_config['task_name'], train_config['ckpt_name'])):
        print('Loading checkpoint as found one')
        encoder.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                        'encoder_' + train_config['ckpt_name']), map_location=device))
        diffusion.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                          'diffusion_' + train_config['ckpt_name']), map_location=device))

    # Specify training parameters
    num_epochs = train_config['num_epochs']
    optimizer = Adam(list(encoder.parameters()) + list(diffusion.parameters()), lr=train_config['lr'])

    # Run training
    for epoch_idx in range(num_epochs):
        losses = []
        for data in tqdm(shapenet_loader):
            optimizer.zero_grad()
            pc = data['pointcloud'].float().to(device)

            # Encode point cloud to latent code
            latent_code, _ = encoder(pc)

            # Compute diffusion loss
            loss = diffusion.get_loss(pc, latent_code)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses),
        ))
        torch.save(encoder.state_dict(), os.path.join(train_config['task_name'],
                                                      'encoder_' + train_config['ckpt_name']))
        torch.save(diffusion.state_dict(), os.path.join(train_config['task_name'],
                                                        'diffusion_' + train_config['ckpt_name']))

    print('Done Training ...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    train(args)