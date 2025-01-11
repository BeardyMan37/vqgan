import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import utils as vutils
from discriminator import Discriminator
from lpips import LPIPS
from vqgan import VQGAN
from utils import load_data, weights_init
import matplotlib.pyplot as plt
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.optim import Adam, lr_scheduler

class label_matching_loss(nn.Module):
    
    def __init__(self, norm='L1'):
        super(label_matching_loss, self).__init__()
        self.norm = norm
        if norm == 'L1':
            self.criterion = nn.L1Loss(reduction='sum')
        elif norm == 'L2':
            self.criterion = nn.MSELoss(reduction='sum')
            
    def forward(self, structured, labels):
        return self.criterion(structured, labels)/len(labels)

class Mse_loss(nn.Module):
    def __init__(self, norm='L2'):
        super(Mse_loss, self).__init__()
        
        if norm == 'L2':
            self.criterion = nn.MSELoss(reduction='sum')
        elif norm == 'L1':
            self.criterion = nn.L1Loss(reduction='sum')
            
    def forward(self, unstructured, labels):
        batch_size = unstructured.size(0)
        num_classes = labels.size(1)
        d = unstructured.size(1)
        labels = labels.to(args.device)
        
        Labels = labels.transpose(0,1).reshape(num_classes, batch_size, 1)
        mask_pos = Labels == 1
        mask_neg = ~mask_pos
        mask_pos_non_zero_sum = torch.sum(mask_pos, (1,2)).reshape(-1,1)
        I = mask_pos_non_zero_sum == 0
        mask_pos_non_zero_sum[I] = 1
        mask_neg_non_zero_sum = torch.sum(mask_neg, (1,2)).reshape(-1,1)
        J = mask_neg_non_zero_sum == 0
        mask_neg_non_zero_sum[J] = 1
        
        class_unstructured_pos = torch.sum(mask_pos * unstructured, 1)/mask_pos_non_zero_sum
        class_unstructured_neg = torch.sum(mask_neg * unstructured, 1)/mask_neg_non_zero_sum
        
        MSE = self.criterion(class_unstructured_pos, torch.zeros_like(class_unstructured_pos).to(args.device))
        MSE += self.criterion(class_unstructured_neg, torch.zeros_like(class_unstructured_neg).to(args.device))
        N = sum(~I) + sum(~J)
        
        return (MSE/N)[0]

class Content_loss(nn.Module):
    
    def __init__(self):
        super(Content_loss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, output, target):
        loss_list = [self.criterion(output[layer], target[layer]) for layer in range(len(output))]
        return sum(loss_list)/len(loss_list)
    
class ImageNet_Norm_Layer_2(nn.Module):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(ImageNet_Norm_Layer_2, self).__init__()
        self.cuda = torch.cuda.is_available()
        dtype = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.mean = Variable(torch.FloatTensor(mean).type(dtype), requires_grad=0)
        self.std = Variable(torch.FloatTensor(std).type(dtype), requires_grad=0)

    def forward(self, input):
        return ((input.permute(0, 2, 3, 1) - self.mean) / self.std).permute(0, 3, 1, 2)

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.norm_layer = ImageNet_Norm_Layer_2()

        vgg = models.vgg19(pretrained=True)
        vgg_feats = vgg.features
        layers = list(vgg_feats.children())

        self.conv1_1 = layers.pop(0)
        layers.pop(0)
        self.conv1_2 = layers.pop(0)
        layers.pop(0)
        self.pool1 = layers.pop(0)

        self.conv2_1 = layers.pop(0)
        layers.pop(0)
        self.conv2_2 = layers.pop(0)
        layers.pop(0)
        self.pool2 = layers.pop(0)

        self.conv3_1 = layers.pop(0)
        layers.pop(0)
        self.conv3_2 = layers.pop(0)
        layers.pop(0)
        self.conv3_3 = layers.pop(0)
        layers.pop(0)
        self.conv3_4 = layers.pop(0)
        layers.pop(0)
        self.pool3 = layers.pop(0)

        self.conv4_1 = layers.pop(0)
        layers.pop(0)
        self.conv4_2 = layers.pop(0)
        layers.pop(0)
        self.conv4_3 = layers.pop(0)
        layers.pop(0)
        self.conv4_4 = layers.pop(0)
        layers.pop(0)
        self.pool4 = layers.pop(0)

        self.conv5_1 = layers.pop(0)
        layers.pop(0)
        self.conv5_2 = layers.pop(0)
        layers.pop(0)
        self.conv5_3 = layers.pop(0)
        layers.pop(0)
        self.conv5_4 = layers.pop(0)
        layers.pop(0)
        self.pool5 = layers.pop(0)

    def forward(self, x, out_keys):
        
        out = {}
        out['in'] = x
        x = self.norm_layer(x)
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        
        return [out[key] for key in out_keys]

class TrainVQGAN:
    def __init__(self, args):
        self.vqgan = VQGAN(args).to(device=args.device)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)
        self.span = torch.tensor(torch.load('checkpoints/vqgan_span.pt')['span']).to(device=args.device)

        # self.usage_codebook = np.zeros(shape=(args.num_codebook_vectors))

        self.prepare_training()

        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))

        return opt_vq, opt_disc

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train(self, args):
        train_dataset = load_data(args)
        steps_per_epoch = len(train_dataset)
        scaler = GradScaler()
        # loss_network = VGG().to(args.device)
        # content_layers = ['r11', 'r21', 'r31']
        # content_loss = Content_loss().to(args.device)
        # dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        # losses, ortho_losses, content_losses, mse_losses, decoder_losses = [], [], [], [], []
        
        for epoch in range(args.epochs):
            # usage_codebook = np.zeros(shape=(args.num_codebook_vectors))
            with tqdm(range(len(train_dataset))) as pbar:
                for i, data in zip(pbar, train_dataset):
                    imgs, labels = data[0], data[1]
                    # content_loss_list, ortho_loss_list, mse_loss_list, decoder_loss_list, gan_loss_list, vq_loss_list = [], [], [], [], [], []
                    self.discriminator.zero_grad()
                    self.opt_disc.zero_grad()
                    with autocast(device_type='cuda', dtype=torch.float16):
                        imgs = imgs.to(device=args.device)
                        # labels = labels.to(device=args.device)
                        with torch.no_grad():
                        #     decoded_images, unstructured, structured = self.vqgan(imgs)
                        # target_step_1 = loss_network(imgs, out_keys=content_layers)
                        # output_step_1 = loss_network(decoded_images, out_keys=content_layers)
                            encoded_images = self.vqgan.encode(imgs)
                            encoded_images_linearized = encoded_images.reshape(*encoded_images.shape[:2], -1)
                            encoded_images_projected = encoded_images_linearized @ self.span @ self.span.T
                            encoded_images_projected_unwrapped = encoded_images_projected.reshape(*encoded_images_projected.shape[:2], *encoded_images.shape[2:])
                            decoded_images = self.vqgan.decode(encoded_images_projected_unwrapped)
                            # decoded_images = self.vqgan(imgs)

                        # ct_loss_step_1 = torch.mean(content_loss(output_step_1, target_step_1))
                        # label_match_loss = label_matching_loss('L1')
                        # ortho_loss_step_1 = torch.mean(label_match_loss(structured, labels))
                        # # MSE_loss = Mse_loss(norm='L2').to(device=args.device)
                        # # mse_loss_step_1 = MSE_loss(unstructured, labels)

                        disc_real = self.discriminator(imgs)
                        disc_fake = self.discriminator(decoded_images.detach())
                        d_loss_real = torch.mean(F.relu(1. - disc_real))
                        d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                        disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch * steps_per_epoch + i,
                                                              threshold=args.disc_start)
                        # gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake) + ct_loss_step_1 + ortho_loss_step_1
                        gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)
                    scaler.scale(gan_loss).backward()
                    scaler.step(self.opt_disc)
                    scaler.update()

                    # gan_loss.backward()
                    # print(self.vqgan.decoder.conv_out.weight.grad.max(), self.discriminator.model[-1].weight.grad.max())

                    # self.opt_disc.step()

                    self.vqgan.zero_grad()
                    self.opt_vq.zero_grad()
                    with autocast(device_type='cuda', dtype=torch.float16):
                        # imgs = imgs.to(device=args.device)
                        # decoded_images, unstructured, structured = self.vqgan(imgs)
                        # target_step_2 = loss_network(imgs, out_keys=content_layers)
                        # output_step_2 = loss_network(decoded_images, out_keys=content_layers)
                        encoded_images = self.vqgan.encode(imgs)
                        encoded_images_linearized = encoded_images.reshape(*encoded_images.shape[:2], -1)
                        encoded_images_projected = encoded_images_linearized @ self.span @ self.span.T
                        encoded_images_projected_unwrapped = encoded_images_projected.reshape(*encoded_images_projected.shape[:2], *encoded_images.shape[2:])
                        decoded_images = self.vqgan.decode(encoded_images_projected_unwrapped)
                        # decoded_images = self.vqgan(imgs)

                        # ct_loss_step_2 = torch.mean(content_loss(output_step_2, target_step_2))
                        # label_match_loss = label_matching_loss('L1')
                        # ortho_loss_step_2 = torch.mean(label_match_loss(structured, labels))
                        # # MSE_loss = Mse_loss(norm='L2').to(device=args.device)
                        # # mse_loss_step_2 = MSE_loss(unstructured, labels)

                        perceptual_loss = self.perceptual_loss(imgs.contiguous(), decoded_images.contiguous())
                        rec_loss = torch.abs(imgs.contiguous() - decoded_images.contiguous())
                        disc_fake = self.discriminator(decoded_images)
                        perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                        perceptual_rec_loss = perceptual_rec_loss.mean()
                        g_loss = -torch.mean(disc_fake)

                        lbd = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                        # vq_loss = perceptual_rec_loss + disc_factor * lbd * g_loss + ct_loss_step_2 + ortho_loss_step_2
                        vq_loss = perceptual_rec_loss + disc_factor * lbd * g_loss
                    # min_encoding_indices_int = min_encoding_indices.cpu()
                    # for j in range(len(min_encoding_indices_int)):
                    #     usage_codebook[min_encoding_indices_int[j]] += 1

                    scaler.scale(vq_loss).backward()
                    scaler.step(self.opt_vq)
                    scaler.update()

                    if i % 1000 == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat((imgs.add(1).mul(0.5)[:4],
                                                          torch.clamp(decoded_images.add(1).mul(0.5), min=0.0, max=1)[
                                                          :4]))
                            vutils.save_image(real_fake_images, os.path.join("results", f"{epoch}_{i}.jpg"), nrow=4)

                    pbar.set_postfix(
                        VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3),
                        lbd=lbd.data.cpu().numpy()
                    )
                    pbar.update(0)
                torch.save(self.vqgan.state_dict(), os.path.join("checkpoints", f"vqgan_epoch_{epoch}.pt"))
                torch.save(self.discriminator.state_dict(),
                           os.path.join("checkpoints", f"discriminator_epoch_{epoch}.pt"))
                # plt.imshow(usage_codebook.reshape(32, 32))
                # plt.savefig(os.path.join("checkpoints", f"codebook_epoch_{epoch}.png"))
                # np.save(os.path.join("checkpoints", f"codebook_epoch_{epoch}.npy"), usage_codebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
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

    train_vqgan = TrainVQGAN(args)
