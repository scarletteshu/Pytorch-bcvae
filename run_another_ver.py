import os
import random
import configs as cfg
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from dataLoad_copy import Dataloader
from bcvae2d_copy import ConditionalVAE

import torch
from torch import Tensor


class RunModel():

    def __init__(self,):
        self.dl = Dataloader()

        # coefficient
        self.MAXEPOCH    = cfg.params['EPOCH']
        self.epoch       = 0
        self.num_iter    = 0
        self.batch_size  = cfg.params['batch_size']
        #self.sch_gamma   = cfg.params['sch_gamma']   #for decay
        self.lr          = cfg.params['lr']
        self.model_gamma = cfg.params['gamma']
        self.max_iter = cfg.params['Capacity_max_iter']
        self.sch_gamma = cfg.params['sch_gamma']  # for decay
        # model
        self.loss = 0
        self.curr_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model = ConditionalVAE(in_channels=1,
                                    latent_dim=6,
                                    gamma=self.model_gamma,
                                    cur_device=self.curr_device,
                                    batch_size=self.batch_size)

        self.model = self.model.to(self.curr_device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.sch_gamma)

    def recordLoss(self, e, loss, recons_loss, kld_loss, states="train"):
        f_tr = open(cfg.params["results_path"] + states + "/loss/" + states[:2] + "_loss_epoch_" + str(e) + ".pkl", "ab+")

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

    def loadModel(self):
        if os.path.isfile(cfg.params['models_path'] + cfg.params['model_name'] + "3.pth.tar"):
            print("model loading......")
            # only for inference: self.model.load_state_dict(torch.load("model.pth.tar"))
            checkpoint = torch.load(cfg.params['models_path'] + cfg.params['model_name'] + "3.pth.tar")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.epoch = checkpoint['epoch']
            self.num_iter = checkpoint['iter']
        else:
            pass

    def run(self):
        # checkpoint loading
        self.loadModel()

        for e in range(self.epoch, self.MAXEPOCH):
            # for k cross validation
            #k = e % 10
            #tr_dataloader = self.dl.train_dataloader(k)

            #train
            self.model.train()
            for i, target_img in enumerate(self.dl.train_loader):
                self.num_iter += 1
                target_img = target_img.to(self.curr_device)
                recons, mu, log_var = self.model(target_img)

                loss, recons_loss, kld = self.model.lossFunc(recons, target_img, mu, log_var, self.num_iter)

                print("lr:{}, ".format(self.optimizer.state_dict()['param_groups'][0]['lr']),
                      "epoch {}, iter {}\n".format(e, self.num_iter),
                      "loss:{}, recons_loss:{}, kld_loss:{}\n".format(loss.item(), recons_loss.item(), kld.item()))

                if i % 100 == 0:
                    self.recordResults(e, recons, "recons")
                    self.recordResults(e, target_img, "origins")

                if i % 50 == 0:
                    self.recordLoss(e, loss.item(), recons_loss.item(), kld.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            torch.save({
                'epoch': e+1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'iter': self.num_iter
            }, cfg.params['models_path'] + cfg.params['model_name'] + "2.pth.tar")
            if (e+1) % 3 == 0:
                self.scheduler.step()

            # validate after every epoch
            self.model.eval()
            self.validate(e)


    def validate(self, e: int):
        loss = []
        recons_loss = []
        kld_loss = []

        with torch.no_grad():
            for i, val_img in enumerate(self.dl.val_loader):
                val_img = val_img.to(self.curr_device)
                val_recons, val_mu, val_log_var = self.model(val_img)

                if i % 200 == 0:
                    self.recordResults(e, val_recons, "recons", "val")
                    self.recordResults(e, val_img, "origins", "val")

                val_loss, val_recons_loss, val_kld = self.model.lossFunc(val_recons, val_img, val_mu, val_log_var, self.num_iter)
                loss.append(val_loss.item())
                recons_loss.append(val_recons_loss.item())
                kld_loss.append(val_kld.item())

            avg_loss = sum(loss)/len(loss)
            avg_recons_loss = sum(recons_loss)/len(recons_loss)
            avg_kld_loss    = sum(kld_loss)/len(kld_loss)

            print("-----------------validating------------------")
            print("epoch{}\tavg_loss:{}, avg_recons_Loss:{}, avg_KLD:{}\n".format(e, avg_loss, avg_recons_loss, avg_kld_loss))
            print("---------------------------------------------")

            self.recordLoss(e, avg_loss, avg_recons_loss, avg_kld_loss,  "val")

    #latent traversal
    def save_traverse(self, sample, fixed_name, vary_dim, val, plot_len):
        save_dir = './results/plot/latent_trav/' + fixed_name + '/' + str(vary_dim) + '_' + str(val) + '.jpg'
        sample = sample.to('cpu').numpy()
        plt.imsave(save_dir, sample, cmap='gray')

    def latent_traverse(self, limit=3, inter=2 / 3, loc=-1):
        if os.path.isfile("fvae.pth.tar"):
            print("model loading......")
            # only for inference: self.model.load_state_dict(torch.load("model.pth.tar"))
            checkpoint = torch.load("fvae.pth.tar")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.model.eval()
        # latents = torch.from_numpy(sample_latent(latents_sizes).astype('float32'))


        dsets_len = self.dl.train_data.__len__()
        rand_idx = random.randint(1, dsets_len - 1)

        random_img = self.dl.train_data.__getitem__(rand_idx).unsqueeze(0)
        random_img = random_img.to(self.curr_device)
        distribution = self.model.encode(random_img)
        mu = distribution[:, :self.model.latent_dim]
        log_var = distribution[:, self.model.latent_dim:]
        random_img_z = self.model.reparameterize(mu, log_var)

        fixed_idx1 = 87040  # square
        fixed_idx2 = 332800  # ellipse
        # fixed_idx3 = 578560 # heart

        fixed_img1 = self.dl.train_data.__getitem__(fixed_idx1).unsqueeze(0)
        fixed_img1 = fixed_img1.to(self.curr_device)
        distribution = self.model.encode(fixed_img1)
        mu = distribution[:, :self.model.latent_dim]
        log_var = distribution[:, self.model.latent_dim:]
        fixed_img_z1 = self.model.reparameterize(mu, log_var)

        fixed_img2 = self.dl.train_data.__getitem__(fixed_idx2).unsqueeze(0)
        fixed_img2 = fixed_img2.to(self.curr_device)
        distribution = self.model.encode(fixed_img2)
        mu = distribution[:, :self.model.latent_dim]
        log_var = distribution[:, self.model.latent_dim:]
        fixed_img_z2 = self.model.reparameterize(mu, log_var)
        """
        fixed_img3 = self.dl.dataset.__getitem__(fixed_idx3)
        fixed_img3 = Variable(cuda(fixed_img3, self.use_cuda), volatile=True).unsqueeze(0)
        fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]
        """

        Z = {'fixed_square': fixed_img_z1, 'fixed_ellipse': fixed_img_z2,
             'random_img': random_img_z}

        interpolation = torch.arange(-limit, limit + 0.1, inter)
        # print(interpolation)

        for key in Z.keys():
            z_ori = Z[key]
            sample = self.model.decode(z_ori).data.squeeze()
            print(sample.shape)
            self.save_traverse(sample, key, 7, 0, interpolation.size)

            # samples.append(sample)
            # samples = []
            for row in range(6):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:  # 一维改变
                    z[:, row] = val
                    print(z)
                    sample = self.model.decode(z).data.squeeze()
                    print(sample.shape)
                    self.save_traverse(sample, key, row, val, interpolation.size)
                    # samples.append(sample)

            # samples = torch.cat(samples, dim=0).cpu()

if __name__ == "__main__":
    r = RunModel()
    print(r.model)
    #r.run()
    r.latent_traverse()
