import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from options import *


class DatasetLoader(Dataset):
    def __init__(self, k, data_type, opt):
        super().__init__()
        if not isinstance(k, int):
            raise ValueError(f"Expected k to be an integer, but got {type(k)}")
        self.k = k
        self.data_type = data_type
        self.opt = opt
        
        # ../imgs_pt/luad (luad | blca| lusc)
        imgs_base_folder = opt.imgs_pt_root
        self.path_data_folder = os.path.join(imgs_base_folder)
        
        # ../datasets_csv/luad (luad | blca| lusc)
        datasets_base_folder = opt.datasets_csv_root
        self.gene_data_folder = os.path.join(datasets_base_folder, 'RNA_Label')
        self.label_data_folder = os.path.join(datasets_base_folder, 'RNA_Label')
        self.matrix_data_folder = os.path.join(datasets_base_folder, 'Edge_Matrix')
        
        self.load_data()

    def load_data(self):
        # Load genomics data
        gene_data_path = os.path.join(self.gene_data_folder, f'{self.k}_{self.data_type}.csv')
        gene_data = pd.read_csv(gene_data_path)
        self.gene_ids = []
        self.gene_base_ids = []
        for slide_id in gene_data['Slide_id'].tolist():
            base_id = slide_id.rsplit('.', 1)[0]
            self.gene_ids.append(slide_id)
            self.gene_base_ids.append(base_id)
        # Top-K Differential Expressed Genes(DEGs): K = 200
        gene_features = gene_data.iloc[:, 1:201].values.reshape(-1, 200, 1).astype(float)
        self.gene_features = torch.FloatTensor(gene_features).requires_grad_()

        # Load adjacency matrix
        edge_matrix_path = os.path.join(self.matrix_data_folder, f'EdgeMatrix_{self.k}.csv')
        edge_matrix_data = pd.read_csv(edge_matrix_path, header=None)
        edge_matrix = np.array(edge_matrix_data).astype(float)
        self.edge_matrix = torch.from_numpy(edge_matrix)

        # Load labels data
        label_data_path = os.path.join(self.label_data_folder, f'{self.k}_{self.data_type}.csv')
        label_data = pd.read_csv(label_data_path, sep=',')
        label_data.columns = label_data.columns.str.strip()
        self.labels = {
            row['Slide_id']: (row['censored'], row['Survival months'], row['discrete_time'])
            for _, row in label_data.iterrows()
        }
        
        # Load pathology data
        self.path_files = []
        self.path_base_ids = []
        for file_name in os.listdir(self.path_data_folder):
            if file_name.endswith('.pt'):
                file_path = os.path.join(self.path_data_folder, file_name)
                base_id = file_name.rsplit('.', 1)[0]
                self.path_files.append(file_path)
                self.path_base_ids.append(base_id)

    #  Return dataset size
    def __len__(self):
        return len(self.gene_ids)

    # Get multimodal data
    def __getitem__(self, idx):
        gene_id = self.gene_ids[idx]
        gene_feature = self.gene_features[idx]
        
        label = self.labels.get(gene_id, (0, 0.0, 0))
        label_tensor = torch.tensor(label)
        
        gene_base_id = self.gene_base_ids[idx]
        path_idx = self.path_base_ids.index(gene_base_id)
        path_file = self.path_files[path_idx]
        path_feature = torch.load(path_file)
        
        return {
            'path_feature': path_feature,
            'gene_feature': gene_feature,
            'edge_matrix': self.edge_matrix,
            'label': label_tensor
        }
    