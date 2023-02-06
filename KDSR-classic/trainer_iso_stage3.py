import os
import utility2
import utility
import torch
from decimal import Decimal
import torch.nn.functional as F
from utils import util
from utils import util2
from collections import OrderedDict
import random
import numpy as np
import torch.nn as nn
import utility3


class Trainer():
    def __init__(self, args, loader, model_TA, my_loss, ckp):
        self.test_res_psnr = []
        self.test_res_ssim = []
        self.args = args
        self.scale = args.scale
        self.loss1= nn.L1Loss()
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model_TA = model_TA
        self.loss = my_loss
        self.optimizer = utility2.make_optimizer(args, self.model_TA)
        self.scheduler = utility2.make_scheduler(args, self.optimizer)
        self.pixel_unshuffle = nn.PixelUnshuffle(self.scale[0])
        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1

        lr = self.args.lr_sr * (self.args.gamma_sr ** ((epoch - self.args.epochs_encoder) // self.args.lr_decay_sr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model_TA.train()

        degrade = util2.SRMDPreprocessing(
            self.scale[0],
            kernel_size=self.args.blur_kernel,
            blur_type=self.args.blur_type,
            sig_min=self.args.sig_min,
            sig_max=self.args.sig_max,
            lambda_min=self.args.lambda_min,
            lambda_max=self.args.lambda_max,
            noise=self.args.noise
        )

        timer = utility2.timer()

        for batch, ( hr, _,) in enumerate(self.loader_train):
            hr = hr.cuda() # b, c, h, w
            timer.tic()
            loss_all = 0
            lr_blur, hr_blur = degrade(hr)  # b, c, h, w
            hr2 = self.pixel_unshuffle(hr)
            sr, _ = self.model_TA(lr_blur, torch.cat([lr_blur,hr2], dim=1))
            # sr,_ = self.model_TA(lr_blur, torch.cat([hr_blur,hr],dim=1))
            loss_all += self.loss1(sr,hr)
            self.optimizer.zero_grad()
            loss_all.backward()
            self.optimizer.step()

            # Remove the hooks before next training phase
            timer.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log(
                    'Epoch: [{:04d}][{:04d}/{:04d}]\t'
                    'Loss [SR loss:{:.3f}]\t'
                    'Time [{:.1f}s]'.format(
                        epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                        loss_all.item(),
                        timer.release(),
                    ))

        self.loss.end_log(len(self.loader_train))

        # save model
        if epoch > self.args.st_save_epoch or (epoch %20 ==0):
            target = self.model_TA.get_model()
            model_dict = target.state_dict()
            torch.save(
                model_dict,
                os.path.join(self.ckp.dir, 'model', 'model_TA_{}.pt'.format(epoch))
            )

            optimzer_dict = self.optimizer.state_dict()
            torch.save(
                optimzer_dict,
                os.path.join(self.ckp.dir, 'optimzer', 'optimzer_TA_{}.pt'.format(epoch))
            )

        target = self.model_TA.get_model()
        model_dict = target.state_dict()
        torch.save(
            model_dict,
            os.path.join(self.ckp.dir, 'model', 'model_TA_last.pt')
        )
        optimzer_dict = self.optimizer.state_dict()
        torch.save(
            optimzer_dict,
            os.path.join(self.ckp.dir, 'optimzer', 'optimzer_TA_last.pt')
        )


    def test(self):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model_TA.eval()
        if self.scale[0]==4:
            sig_list=[1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2]
        elif self.scale[0]==2:
            sig_list=[0.9,1,1.1,1.2,1.3,1.4,1.5,1.6]
        timer_test = utility.timer()

        with torch.no_grad():

            for idx_data, d in enumerate(self.loader_test):
                for idx_scale, scale in enumerate(self.scale):
                    d.dataset.set_scale(idx_scale)
                    eval_psnr = 0
                    eval_ssim = 0
                    for idx, (hr0, filename) in enumerate(d):
                        for sig in sig_list:
                            degrade = util2.SRMDPreprocessing(
                                self.scale[0],
                                kernel_size=self.args.blur_kernel,
                                blur_type=self.args.blur_type,
                                sig=sig,
                                lambda_1=self.args.lambda_1,
                                lambda_2=self.args.lambda_2,
                                theta=self.args.theta,
                                noise=self.args.noise
                            )

                            hr = hr0.cuda()                      # b, c, h, w
                            hr = self.crop_border(hr, scale)

                            # inference
                            timer_test.tic()
                            lr_blur, hr_blur = degrade(hr, random=False)
                            hr2 = self.pixel_unshuffle(hr)
                            sr = self.model_TA(lr_blur, torch.cat([lr_blur, hr2], dim=1))
                            timer_test.hold()

                            sr = utility.quantize(sr, self.args.rgb_range)
                            hr = utility.quantize(hr, self.args.rgb_range)

                            # metrics
                            # eval_psnr += utility.calc_psnr(
                            #     sr, hr, scale, self.args.rgb_range,
                            #     benchmark=d.dataset.benchmark
                            # )
                            eval_psnr += utility3.calc_psnr(
                                sr, hr, scale
                            )
                            eval_ssim += utility.calc_ssim(
                                (sr * 255).round().clamp(0, 255), (hr * 255).round().clamp(0, 255), scale,
                                benchmark=d.dataset.benchmark
                            )

                        # save results
                        if self.args.save_results:
                            save_list = [sr]
                            filename = filename[0]
                            self.ckp.save_results(filename, save_list, scale)

                    # if len(self.test_res_psnr) > 10:
                    #     self.test_res_psnr.pop(0)
                    #     self.test_res_ssim.pop(0)
                    # self.test_res_psnr.append(eval_psnr / len(d) /len(sig_list))
                    # self.test_res_ssim.append(eval_ssim / len(d)/len(sig_list))

                    self.ckp.log[-1, idx_scale] = eval_psnr / len(d)/len(sig_list)
                    self.ckp.write_log(
                        '[Epoch {}---{} x{}]\tPSNR: {:.3f} SSIM: {:.4f}'.format(
                            self.args.resume,
                            self.args.data_test,
                            scale,
                            eval_psnr / len(d) /len(sig_list),
                            eval_ssim / len(d) /len(sig_list)
                        ))
    
    def crop_border(self, img_hr, scale):
        b, c, h, w = img_hr.size()

        img_hr = img_hr[:, :, :int(h//scale*scale), :int(w//scale*scale)]

        return img_hr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >=  self.args.epochs_sr