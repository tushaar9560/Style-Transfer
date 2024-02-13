import os
import time
import numpy as np
import sys
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from utils import load_image, normalize_batch, gram_matrix
from logger import logging
from exception import CustomException
from components.transformer_net import TransformerNet
from components.vgg  import Vgg16

class StyleTransferTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.transformer = TransformerNet().to(self.device)
        self.vgg = Vgg16(requires_grad = False).to(self.device)
        self.optimizer = Adam(self.transformer.parameters(), args.lr)
        self.mse_loss = torch.nn.MSELoss()

    def train(self):
        try:
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)

            transform = transforms.Compose([
                transforms.Resize(self.args.image_size),
                transforms.CenterCrop(self.args.image_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255))
            ])
            logging.info("Dataset Loading....")
            train_dataset = datasets.ImageFolder(self.args.dataset, transform)
            train_loader = DataLoader(train_dataset, batch_size = self.args.batch_size)
            logging.info("Dataset Loaded....")

            style_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255))
            ])

            logging.info("Loading Style image")
            style = load_image(self.args.style_image, size = self.args.style_size)
            style = style_transform(style)
            style = style.repeat(self.args.batch_size, 1, 1, 1).to(self.device)
            features_style = self.vgg(normalize_batch(style))
            gram_style = [gram_matrix(y) for y in features_style]

            logging.info("Training Started....")
            for e in range(self.args.epochs):
                self.transformer.train()
                agg_content_loss = 0.
                agg_style_loss = 0.
                count = 0
                for batch_id, (x, _) in enumerate(train_loader):
                    n_batch = len(x)
                    count += n_batch
                    self.optimizer.zero_grad()

                    x = x.to(self.device)
                    y = self.transformer(x)

                    y = normalize_batch(y)
                    x = normalize_batch(x)

                    features_y = self.vgg(y)
                    features_x = self.vgg(x)

                    content_loss = self.args.content_weight * self.mse_loss(features_y.relu2_2, features_x.relu2_2)

                    style_loss = 0.
                    for ft_y, gm_s in zip(features_y, gram_style):
                        gm_y = gram_matrix(ft_y)
                        style_loss += self.mse_loss(gm_y, gm_s[:n_batch, :, :])
                    style_loss *= self.args.style_weight

                    total_loss = content_loss + style_loss
                    total_loss.backward()
                    self.optimizer.step()

                    agg_content_loss += content_loss.item()
                    agg_style_loss += style_loss.item()

                    if (batch_id + 1) % self.args.log_interval == 0:
                        mseg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                                time.ctime(), e+1, count, len(train_dataset),
                                agg_content_loss/(batch_id+1), agg_style_loss/(batch_id+1),
                                (agg_content_loss + agg_style_loss) / (batch_id+1)
                        )
                    logging.info(mseg)

                    if self.args.checkpoint_model_dir is not None and (batch_id + 1) % self.args.checkpoint_interval == 0:
                        self.transformer.eval().cpu()
                        ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id+1) + ".pth"
                        ckpt_model_path = os.path.join(self.args.checkpoint_model_dir, ckpt_model_filename)
                        torch.save(self.transformer.state_dict(), ckpt_model_path)
                        self.transformer.to(self.device).train()

            logging.info("Saving model")
            self.transformer.eval().cpu()
            save_model_filename = "epoch_" + str(self.args.epochs)+ "_" + str(time.ctime()).replace(" ", "_") + "_" 
            + str(self.args.content_weight) + "_" + str(self.args.style_weight) + ".model"
            save_model_path = os.path.join(self.args.save_model_dir, save_model_filename)
            torch.save(self.transformer.state_dict(), save_model_path)

            logging.info(f"Model training complete and saved at {save_model_path}")

        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
