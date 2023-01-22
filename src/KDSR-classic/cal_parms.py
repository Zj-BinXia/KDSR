def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

from option import args
from model_ST.blindsr import make_model
# args.n_feats=64
args.scale=[4]
net = make_model(args)
print("KDSRs-M parameters (M):",count_param(net))

args.scale=[4]
args.n_feats=128 
args.n_blocks=28 
args.n_resblocks=5
net = make_model(args)
print("KDSRs-L parameters (M):",count_param(net))
