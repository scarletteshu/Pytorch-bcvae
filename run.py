import os
import configs as cfg
import numpy as np
import pickle as pkl
from dataLoad import Dataloader

import torch
from torch import Tensor
import torch.nn.functional as F
from bcvae2d import ConditionalVAE


class RunModel():

    def __init__(self,):
        self.dl = Dataloader()
        self.dl.datasplit()
        # coefficient
        self.MAXEPOCH = cfg.params['EPOCH']
        self.num_iter = 0
        self.epoch = 0
        self.batch_size = cfg.params['batch_size']
        self.sch_gamma = cfg.params['sch_gamma']   #for decay
        self.lr = cfg.params['lr']
        # model
        self.curr_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model = ConditionalVAE(1, 6)
        self.model = self.model.cuda(self.curr_device)
        self.model.C_max = self.model.C_max.to(self.curr_device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=self.sch_gamma)

    def reconsLoss(self, recons, input):
        loss = F.mse_loss(recons, input, size_average=False).div(self.batch_size)
        return loss

    def kldLoss(self, mu, log_var):
        loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return loss

    def Loss(self, recons_loss, kld_loss, kld_weight):
        """
        c [0, 25], linearly increasing
        as iteration continue, loss will be up to gamma*0.0008680*|kldloss-25| + recons_loss
        """
        C = torch.clamp(self.model.C_max / self.model.C_stop_iter * self.num_iter, 0, self.model.C_max.data[0])
        loss = recons_loss + self.model.gamma * kld_weight * (kld_loss - C).abs()
        return loss

    def recordLoss(self, e, loss, recons_loss, kld_loss,states="train"):
        f_tr = open(cfg.params["results"] + states + "/loss/" + states[:2] + "_loss_epoch_" + str(e) + ".pkl", "ab+")

        pkl.dump(np.array([
            loss,
            recons_loss,
            kld_loss,
        ]), f_tr)

    def recordResults(self, e: int, x: Tensor, type: str, states="train"):
        # exp: "./results/train/latents/tr_recons_epoch_0.pkl"
        f = open(cfg.params['results_path'] + states + "/" + type + "/" + states[:2] + "_" + type + "_epoch_" + str(e) + ".pkl", "ab+")
        pkl.dump(x.squeeze(1).to('cpu').detach().numpy(), f)
        f.close()

    def train(self):
        # checkpoint loading
        if os.path.isfile(cfg.params['models_path'] + cfg.params['model_name'] + ".pth.tar"):
            print("model loading......")
            # only for inference: self.model.load_state_dict(torch.load("model.pth.tar"))
            checkpoint = torch.load(cfg.params['models_path'] + cfg.params['model_name'] + ".pth.tar")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.epoch = checkpoint['epoch']
            loss = checkpoint['loss']

        for e in range(self.epoch, self.MAXEPOCH):
            # for k cross validation
            k = e % 10
            tr_dataloader = self.dl.train_dataloader(k)
            # print("lr:{}".format(self.optim.param_groups[0]['lr']))

            self.model.train()
            for i, real_img in enumerate(tr_dataloader):
                self.num_iter += 1
                real_img = real_img.to(self.curr_device)
                results  = self.model.forward(real_img)
                # results: recons_img, input_img, mu, log_var, latent z
                # loss
                recons_loss = self.reconsLoss(results[0], results[1])
                kld_loss    = self.kldLoss(results[2], results[3])
                loss = self.Loss(recons_loss, kld_loss, self.batch_size / self.dl.num_train_imgs)

                print("epoch {}, iter {}\n".format(e, self.num_iter),
                      "loss:{}, recons_loss:{}, kld_loss:{}\n".format(loss.item(), recons_loss.item(), kld_loss.item()))

                if i % 100 == 0:
                    self.recordResults(e, results[0], "recons")
                    self.recordResults(e, results[1], "origins")
                    self.recordResults(e, results[4], "latents")

                if i % 50 == 0:
                    self.recordLoss(e, loss.item(), recons_loss.item(), kld_loss.item())

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            torch.save({
                'epoch': e+1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss': loss,
            }, cfg.params['models_path'] + cfg.params['model_name'] + ".pth.tar")

            self.model.eval()
            # validate after every epoch
            self.validate(e, k)
            # decay for every 10 epoch
            if e % 10 == 0:
                self.scheduler.step()

    def validate(self, e: int, k: int):
        loss = []
        recons_loss = []
        kld_loss = []
        val_dataloader = self.dl.val_dataloader(k)

        with torch.no_grad():
            for i, val_img in enumerate(val_dataloader):
                val_img = val_img.to(self.curr_device)
                val_results = self.model.forward(val_img)

                if i % 200 == 0:
                    self.recordResults(e, val_results[0], "recons", "val")
                    self.recordResults(e, val_results[1], "origins", "val")
                    self.recordResults(e, val_results[4], "latents", "val")

                val_recons_loss = self.reconsLoss(val_results[0], val_results[1])
                val_kld_loss    = self.kldLoss(val_results[2]   , val_results[3])
                val_loss = self.Loss(val_recons_loss, val_kld_loss, self.batch_size / self.dl.num_val_imgs)

                loss.append(val_loss.item())
                recons_loss.append(val_recons_loss.item())
                kld_loss.append(val_kld_loss.item())

            avg_loss = sum(loss)/len(loss)
            avg_recons_loss = sum(recons_loss)/len(recons_loss)
            avg_kld_loss    = sum(kld_loss)/len(kld_loss)

            print("-----------------validating------------------")
            print("epoch{}\tavg_loss:{}, avg_recons_Loss:{}, avg_KLD:{}\n".format(e, avg_loss, avg_recons_loss, avg_kld_loss))
            print("---------------------------------------------")

            self.recordLoss(e, avg_loss, avg_recons_loss, avg_kld_loss,  "val")

    """
    def sample_images(self):

        test_input, test_label = next(iter(self.dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels=test_label)

        vutils.save_image(recons.data, f"{self.log_dir}/recons_{self.cur_epoch}".png,normalize=True, nrow=12)

        #vutils.save_image(recons.data,
        #                  f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                  f"recons_{self.logger.name}_{self.current_epoch}.png",
        #                  normalize=True,
        #                  nrow=12)

        # vutils.save_image(test_input.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"real_img_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        try:
            samples = self.model.sample(144, self.curr_device, labels=test_label)
            vutils.save_image(samples.cpu().data,
                              f"{self.log_dir}/recons_{self.current_epoch}.png",
                              normalize=True,
                              nrow=12)
        except:
            pass

        del test_input, recons  # , samples
        """


if __name__ == "__main__":
    run = RunModel()
    print(run.model)
    run.train()
