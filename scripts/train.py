import pickle
from tqdm import tqdm
from random import random
from scripts.test import test
from torch.backends import cudnn
from torch.utils.data import DataLoader

from models.network import *
from utils.utils_data import *
from utils.utils_analysis import *


def collate_fn(batch):
    path_features_batch = torch.stack([item['path_feature'] for item in batch])
    gene_features = torch.stack([item['gene_feature'] for item in batch])
    edge_matrix = torch.stack([item['edge_matrix'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    return {
        'path_feature': path_features_batch,
        'gene_feature': gene_features,
        'edge_matrix': edge_matrix,
        'label': labels
    }


def train(opt, device, k):
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(2024)
    torch.manual_seed(2024)
    random.seed(2024)

    model = define_net(opt, k)
    optimizer = define_optimizer(opt, model)
    scheduler = define_scheduler(opt, optimizer)
    print(model)
    print("Number of Trainable Parameters: %d" % count_parameters(model))
    print("Activation Type:", opt.act_type)
    print("Optimizer Type:", opt.optimizer_type)
    print("Regularization Type:", opt.reg_type)

    train_loader = DataLoader(DatasetLoader(k, 'train', opt), batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
    metric_logger = {'train': {'loss': [], 'pvalue': [], 'cindex': [], 'surv_acc': []},
                     'test': {'loss': [], 'pvalue': [], 'cindex': [], 'surv_acc': []}}

    for epoch in tqdm(range(opt.epoch_count, opt.niter + opt.niter_decay + 1)):

        model.train()
        loss_epoch = 0
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
        
        for batch_idx, batch_data in enumerate(train_loader):
            path_feature = batch_data['path_feature'].to(device)
            edge_matrix = batch_data['edge_matrix'].to(device)
            gene_feature = batch_data['gene_feature'].to(device)
            
            num_nodes = 200      # Top-K Differential Expressed Genes(DEGs): K = 200
            node_feature_dim = 1
            batch_size = gene_feature.size(0)
            gene_feature = gene_feature.view(batch_size, num_nodes, node_feature_dim)

            labels = batch_data['label'].to(device)
            censor = labels[:, 0] if "surv" in opt.exp_name else None
            survtime = labels[:, 1] if "surv" in opt.exp_name else None
            Yield = labels[:, 2] if "surv" in opt.exp_name else None

            pred, Surv, Y_hatch = model(path=path_feature, omic=gene_feature, edge=edge_matrix)
            
            loss_nll = NLLSurvLoss(alpha=0.4)
            loss_surv = loss_nll(hazards=pred, S=Surv, Y=Yield, c=censor) if opt.exp_name == "surv" else 0
            loss = opt.lambda_surv * loss_surv

            loss_epoch += loss.data.item()
            # print(f"loss_epoch: {loss_epoch}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if opt.exp_name == "surv":
                risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
                censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
                survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))

            if opt.verbose > 0 and opt.print_every > 0 and (batch_idx % opt.print_every == 0 or batch_idx + 1 == len(train_loader)):
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(epoch + 1, opt.niter + opt.niter_decay, batch_idx + 1, len(train_loader), loss.item()))

        scheduler.step()
        # lr = optimizer.param_groups[0]['lr']
        # print('learning rate = %.7f' % lr)

        if opt.measure or epoch == (opt.niter + opt.niter_decay - 1):
            loss_epoch /= len(train_loader)
            cindex_epoch = CIndex_lifeline(risk_pred_all, censor_all, survtime_all) if opt.exp_name == 'surv' else None
            pvalue_epoch = cox_log_rank(risk_pred_all, censor_all, survtime_all) if opt.exp_name == 'surv' else None
            surv_acc_epoch = accuracy_cox(risk_pred_all, censor_all) if opt.exp_name == 'surv' else None
            loss_test, cindex_test, pvalue_test, surv_acc_test, pred_test = test(opt, model, k, device, data_type='val')

            metric_logger['train']['loss'].append(loss_epoch)
            metric_logger['train']['cindex'].append(cindex_epoch)
            metric_logger['train']['pvalue'].append(pvalue_epoch)
            metric_logger['train']['surv_acc'].append(surv_acc_epoch)

            metric_logger['test']['loss'].append(loss_test)
            metric_logger['test']['cindex'].append(cindex_test)
            metric_logger['test']['pvalue'].append(pvalue_test)
            metric_logger['test']['surv_acc'].append(surv_acc_test)

            with open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, f'{opt.model_name}_{k}_{epoch}_test.pkl'), 'wb') as f:
                pickle.dump(pred_test, f)
            if opt.verbose > 0:
                if opt.exp_name == 'surv':
                    print(f'[Train] Loss: {loss_epoch:.4f}, C-Index: {cindex_epoch:.4f}')
                    print(f'[Test] Loss: {loss_test:.4f}, C-Index: {cindex_test:.4f}\n')

    return model, optimizer, metric_logger
