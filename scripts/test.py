from train import *
from utils.utils_data import *
from utils.utils_models import *
from utils.utils_analysis import *
from torch.utils.data import DataLoader


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


def test(opt, model, k, device, data_type):
    model.eval()

    test_loader = DataLoader(DatasetLoader(k, data_type, opt), batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])

    probs_all, gt_all = None, np.array([])
    loss_test = 0

    for batch_idx, batch_data in enumerate(test_loader):
        path_feature = batch_data['path_feature'].to(device)
        edge_matrix = batch_data['edge_matrix'].to(device)
        gene_feature = batch_data['gene_feature'].to(device)
        
        num_nodes = 200  # Top-K Differential Expressed Genes(DEGs): K = 200
        node_feature_dim = 1
        batch_size = gene_feature.size(0)
        gene_feature = gene_feature.view(batch_size, num_nodes, node_feature_dim)

        labels = batch_data['label'].to(device)
        censor = labels[:, 0] if "surv" in opt.exp_name else None
        survtime = labels[:, 1] if "surv" in opt.exp_name else None
        Yield = labels[:, 2] if "surv" in opt.exp_name else None

        with torch.no_grad():
            pred, Surv, Y_hatch = model(path=path_feature, omic=gene_feature, edge=edge_matrix)

        loss_nll = NLLSurvLoss(alpha=0.4)
        loss_surv = loss_nll(hazards=pred, S=Surv, Y=Yield, c=censor) if opt.exp_name == "surv" else 0
        loss = opt.lambda_surv * loss_surv
        loss_test += loss.data.item()

        if opt.exp_name == "surv":
            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))    # Logging Information
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))        # Logging Information
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))  # Logging Information

    loss_test /= len(test_loader)
    cindex_test = CIndex_lifeline(risk_pred_all, censor_all, survtime_all) if opt.exp_name == 'surv' else None
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all) if opt.exp_name == 'surv' else None
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all) if opt.exp_name == 'surv' else None
    pred_test = [risk_pred_all, survtime_all, censor_all, probs_all]

    return loss_test, cindex_test, pvalue_test, surv_acc_test, pred_test
