import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
from discriminator import Discriminator
from vqgan import VQGAN
from utils import load_data
import matplotlib.pyplot as plt
from torch import autocast
from sklearn.decomposition import PCA

class CalculateSpanOfVQGAN:
    def __init__(self, args):
        self.vqgan = VQGAN(args).to(device=args.device)
        self.vqgan.load_state_dict(
            torch.load("checkpoints/vqgan_epoch_1.pt"))
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.load_state_dict(
            torch.load("checkpoints/discriminator_epoch_1.pt"))

        self.calulateSpan(args)

    def calulateSpan(self, args):
        train_dataset = load_data(args)
        avg_representation = torch.zeros((256, 256)).to(device=args.device)
        avg_count = 0      
        with tqdm(range(len(train_dataset))) as pbar:
            for i, data in zip(pbar, train_dataset):
                imgs, _ = data[0], data[1]
                with autocast(device_type='cuda', dtype=torch.float16):
                    imgs = imgs.to(device=args.device)
                    with torch.no_grad():
                        encoded_images = self.vqgan.encode(imgs)
                        encoded_images_flattened = encoded_images.view(4, 256, -1)
                        outer_product = torch.einsum('bij,bik->bijk', encoded_images_flattened, encoded_images_flattened)
                        averaged_images = outer_product.mean(dim=(0, 1))
                        avg_count += 4
                        alpha = 4/avg_count
                        avg_representation = (1 - alpha) * avg_representation + alpha * averaged_images

        pca = PCA(n_components=216)
        pca_result = pca.fit_transform(avg_representation.cpu().numpy())
        print(pca_result.shape)
        print(pca_result)
        torch.save({'span': pca_result}, os.path.join("checkpoints", f"vqgan_span_216.pt"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN Span")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024,
                        help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=4, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=0.2, help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1.0,
                        help='Weighting factor for perceptual loss.')

    args = parser.parse_args()

    CalculateSpanOfVQGAN(args)