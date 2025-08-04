import pickle
import logging

from test import test
from train import train
from utils.utils_data import *

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

opt = parse_args()
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
print("Using device:", device)

if not os.path.exists(opt.checkpoints_dir):
    os.makedirs(opt.checkpoints_dir)
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name)):
    os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name))
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)):
    os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name))

results = []

# Training & Testing
for k in range(0, 4):
    print("*******************************************")
    print("************** SPLIT (%d/%d) **************" % (k, 4))
    print("*******************************************")

    checkpoint_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, f'{opt.model_name}_{k}_pred_train.pkl')
    if os.path.exists(checkpoint_path):
        print("Train-Test Split already made.")
        continue

    model, optimizer, metric_logger = train(opt, device, k)
    
    loss_train, cindex_train, pvalue_train, surv_acc_train, pred_train = test(opt, model, k, device, data_type='train')
    loss_test, cindex_test, pvalue_test, surv_acc_test, pred_test = test(opt, model, k, device, data_type='val')

    if opt.exp_name == 'surv':
        print("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
        logging.info("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
        print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
        logging.info("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
        results.append(cindex_test)

    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        model_state_dict = model.module.cpu().state_dict()
    else:
        model_state_dict = model.cpu().state_dict()
    
    torch.save({
        'split': k,
        'opt': opt,
        'epoch': opt.niter + opt.niter_decay,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metric_logger
        }, os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, f'{opt.model_name}_{k}.pt'))

    pickle.dump(pred_train, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, f'{opt.model_name}_{k}_train.pkl'), 'wb'))
    pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, f'{opt.model_name}_{k}_test.pkl'), 'wb'))

print('Split Results:', results)
print("Average:", np.array(results).mean())
pickle.dump(results, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, f'{opt.model_name}_results.pkl'), 'wb'))
