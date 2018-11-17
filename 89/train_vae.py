import argparse
import logging
import os

import torch
import torchvision.datasets as datasets
from torch.optim import Adam, SGD
from torchvision import transforms

#from homework.dcgan import DCGenerator, DCDiscriminator
#from homework.dcgan import DCGANTrainer
from homework.vae import vae
from homework.vae import trainer as tr


def get_config():
    k = 256
    parser = argparse.ArgumentParser(description='Training VAE on FashionMNIST')

    parser.add_argument('--log-root', type=str, default='../logs')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--log-name', type=str, default='train_vae.log')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--batch-size', type=int, default=k,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train ')
    parser.add_argument('--image-size', type=int, default=28,
                        help='size of images to generate')
    parser.add_argument('--n_show_samples', type=int, default=8)
    parser.add_argument('--show_img_every', type=int, default=10)
    parser.add_argument('--log_metrics_every', type=int, default=2560 // k)
    config = parser.parse_args()
    config.cuda = not config.no_cuda and torch.cuda.is_available()

    return config


def main():
    config = get_config()
    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_root,
                                             config.log_name)),
            logging.StreamHandler()],
        level=logging.INFO)

    transform = transforms.Compose([transforms.Scale(config.image_size), transforms.ToTensor()])#,
                                    # transforms.Normalize((0., 0., 0.), (255., 255., 255.))])
    
    train_dataset = datasets.FashionMNIST(root=config.data_root,
                                     download=True,
                                     train=True,
                                     transform=transform)
    test_dataset = datasets.FashionMNIST(root=config.data_root,
                                    download=True,
                                    train=False,
                                    transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                             num_workers=4, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True,
                                             num_workers=4, pin_memory=True)

    model = vae.VAE()

    trainer = tr.Trainer(model=model,
                         train_loader=train_dataloader,
                         test_loader=test_dataloader,
                         loss_function=vae.loss_function,
                         optimizer=Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999)))
                         # optimizer=SGD(model.parameters(), lr=0.001, momentum=0.5))

    for epoch in range(config.epochs):
        trainer.train(epoch, config.log_metrics_every)
        trainer.test(epoch, config.batch_size, config.log_metrics_every)
        trainer.save(os.path.join('ckpt', f'vae_epoch_{epoch}.pt'))


if __name__ == '__main__':
    main()
