import yaml
import torch
import os
import open3d as o3d
from models.encoders.pointnet import PointNetEncoder
from scheduler.linear_noise_scheduler import LinearNoiseScheduler, DiffusionPoint, PointwiseNet

def load_model_weights(encoder_path, diffusion_path, model_config, diffusion_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the models
    encoder = PointNetEncoder(zdim=model_config['latent_dim']).to(device)
    pointwise_net = PointwiseNet(point_dim=3, context_dim=model_config['latent_dim'],
                                 residual=model_config['residual']).to(device)
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )
    diffusion = DiffusionPoint(net=pointwise_net, scheduler=scheduler).to(device)

    # Load the checkpoints
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    diffusion.load_state_dict(torch.load(diffusion_path, map_location=device))

    return encoder, diffusion, device

def sample_point_cloud(encoder, diffusion, num_points=2048, num_samples=10):
    device = next(encoder.parameters()).device

    encoder.eval()
    diffusion.eval()

    with torch.no_grad():
        latent_vectors = torch.randn(num_samples, encoder.zdim).to(device)

        generated_point_clouds = diffusion.sample(num_points=num_points, context=latent_vectors)

    return generated_point_clouds

def save_point_clouds_as_ply(point_clouds, file_prefix):
    for i, pc in enumerate(point_clouds):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        o3d.io.write_point_cloud(f"{file_prefix}_{i}.ply", pcd)

def visualize_point_clouds(point_clouds):
    vis_list = []
    for pc in point_clouds:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        vis_list.append(pcd)
    o3d.visualization.draw_geometries(vis_list)

def main():
    config_path = 'config/default.yaml'
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    model_config = config['model_params']
    diffusion_config = config['diffusion_params']
    encoder_path = os.path.join(config['train_params']['task_name'], 'encoder_' + config['train_params']['ckpt_name'])
    diffusion_path = os.path.join(config['train_params']['task_name'],
                                  'diffusion_' + config['train_params']['ckpt_name'])

    # Load the models with weights
    encoder, diffusion, device = load_model_weights(encoder_path, diffusion_path, model_config, diffusion_config)

    # Sample point clouds
    generated_point_clouds = sample_point_cloud(encoder, diffusion, num_points=2048, num_samples=10)

    print("Generated Point Clouds: ", generated_point_clouds.shape)

    # Save point clouds as PLY files
    save_point_clouds_as_ply(generated_point_clouds.cpu().numpy(), "generated_point_cloud")

    # Visualize point clouds
    visualize_point_clouds(generated_point_clouds.cpu().numpy())

if __name__ == '__main__':
    main()
