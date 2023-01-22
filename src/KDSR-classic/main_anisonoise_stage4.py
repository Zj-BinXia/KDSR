from option import args
import torch
import utility
import data
import model_TA
import model_ST
import loss
from trainer_anisonoise_stage4 import Trainer


# def count_param(model):
#     param_count = 0
#     for param in model.parameters():
#         param_count += param.view(-1).size()[0]
#     return param_count

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    if checkpoint.ok:
        loader = data.Data(args)
        model_TA = model_TA.Model(args, checkpoint)
        model_ST = model_ST.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader,  model_ST, model_TA, loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()



        checkpoint.done()