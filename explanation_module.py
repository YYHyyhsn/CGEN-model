import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import config
from model.data import CIFData, collate_pool
from model.cgcnn import CrystalGraphConvNet
from torch.utils.data import DataLoader
from pymatgen.core.structure import Structure
from model.GNNExplainer import Explainer, GNNExplainer
from torch.autograd import Variable
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import matplotlib.cm as cm
import sys
import io

# Set stdout encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class Normalizer(object):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor=None):
        if tensor is None:
            self.mean = 0
            self.std = 1
        else:
            self.mean = torch.mean(tensor)
            self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    return torch.mean(torch.abs(target - prediction))


def load_model(model_path, device='cpu'):
    """Load model using the same strategy as in predict.py"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    model_args = checkpoint.get('args', {})

    orig_atom_fea_len = model_args.get('orig_atom_fea_len',
                                       checkpoint.get('orig_atom_fea_len', 92))
    nbr_fea_len = model_args.get('nbr_fea_len',
                                 checkpoint.get('nbr_fea_len', 41))
    atom_fea_len = model_args.get('atom_fea_len', 64)
    n_conv = model_args.get('n_conv', 3)
    h_fea_len = model_args.get('h_fea_len', 128)
    n_h = model_args.get('n_h', 1)
    classification = model_args.get('classification', False)

    model = CrystalGraphConvNet(
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        atom_fea_len=atom_fea_len,
        n_conv=n_conv,
        h_fea_len=h_fea_len,
        n_h=n_h,
        classification=classification
    )

    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    normalizer = Normalizer()
    if 'normalizer' in checkpoint:
        normalizer.load_state_dict(checkpoint['normalizer'])
        print(f"Normalizer loaded successfully: mean={normalizer.mean:.4f}, std={normalizer.std:.4f}")
    else:
        print("Warning: Normalizer not found, using default normalizer")

    print(f"Model loaded successfully: {model_path}")
    return model, normalizer


def run_gnn_explainer_and_visualize(model, dataset, device):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_pool)
    model.eval()
    feature_importance = [[] for _ in range(92)]

    for i, (input, target, batch_cif_ids) in enumerate(tqdm(loader, desc="Explaining samples")):
        input_var = (
            Variable(input[0].to(device)),
            Variable(input[1].to(device)),
            input[2],
            input[3]
        )

        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=config['explain_epochs']),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type=None,
            model_config=dict(
                mode='regression',
                task_level='node',
                return_type='raw',
            )
        )

        kwargs = {'nbr_fea_idx': input_var[2], 'crystal_atom_idx': input_var[3]}
        explanation = explainer(input_var[0], input_var[1], index=None, **kwargs)
        node_mask = explanation.node_mask.cpu().detach().numpy()

        for node in node_mask:
            for idx, value in enumerate(node):
                if value != 0:
                    feature_importance[idx].append(value)

    feature_names = ['La and Ac families', 'Group number', 'Period number', 'Electronegativity',
                     'Covalent radius', 'Valence electrons', 'First ionization energy',
                     'Electron affinity', 'Block', 'Atomic volume']
    feature_intervals = [
        (0, 0), (1, 18), (19, 25), (26, 35), (36, 45),
        (46, 57), (58, 67), (68, 77), (78, 81), (82, 91)
    ]

    importance_scores = []
    for start, end in feature_intervals:
        scores = []
        for i in range(start, end + 1):
            values = feature_importance[i]
            if values:
                scores.append(np.sum(values) / len(values))
            else:
                scores.append(0.0)
        importance_scores.append(scores)

    csv_data = {}
    for name, score_list in zip(feature_names, importance_scores):
        csv_data[name] = score_list
    df_csv = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in csv_data.items()]))
    df_csv.to_csv("feature_importance_values.csv", index_label="Subfeature Index")
    print("Feature importance values saved to feature_importance_values.csv")

    max_length = max(len(scores) for scores in importance_scores)
    heatmap_data = np.full((max_length, len(feature_names)), np.nan)
    for i, scores in enumerate(importance_scores):
        heatmap_data[:len(scores), i] = scores

    df = pd.DataFrame(heatmap_data, columns=feature_names, index=[f"{j + 1}" for j in range(max_length)])
    df_transposed = df.T

    plt.figure(figsize=(14, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    ax = sns.heatmap(df_transposed, annot=True, cmap='RdYlGn_r', cbar=True, linewidths=0, fmt='.2f')
    plt.xlabel("Index", fontsize=14)

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor('black')

    plt.tight_layout()
    plt.savefig('feature_heatmap.svg', format='svg')
    plt.show()

    sorted_values = []
    for feature in df.columns:
        for idx, value in df[feature].dropna().items():
            sorted_values.append((value, feature, idx))
    sorted_values = sorted(sorted_values, key=lambda x: x[0], reverse=True)[:20]

    print("Top 20 most important features, values, and indices:")
    for rank, (value, feature, idx) in enumerate(sorted_values, 1):
        print(f"Rank {rank}: Feature: {feature}, Value: {value:.4f}, Index: {idx}")


def visualize_atom_importance(node_weights, node_names, cif_id):
    total_weight = sum(node_weights)
    normalized_weights = [weight / total_weight for weight in node_weights]

    sorted_idx = np.argsort(normalized_weights)[::-1]
    sorted_node_names = np.array(node_names)[sorted_idx]
    sorted_normalized_weights = np.array(normalized_weights)[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    cmap = cm.RdYlGn_r
    norm = plt.Normalize(vmin=min(sorted_normalized_weights), vmax=max(sorted_normalized_weights))
    colors = cmap(norm(sorted_normalized_weights))

    bars = ax.barh(sorted_node_names, sorted_normalized_weights, align='center', color=colors)
    plt.yticks(fontname='Times New Roman', fontsize=18)
    plt.xlabel('Atom Importance', fontname='Times New Roman', fontsize=18)
    plt.ylabel('Atom', fontname='Times New Roman', fontsize=18)
    plt.gca().invert_yaxis()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax)

    path = f'{cif_id}_atom_importance.svg'
    plt.tight_layout()
    plt.savefig(path)
    print(f"Atom importance plot saved to: {path}")
    plt.show()

    print(f"\nAtom importance for {cif_id}:")
    for name, weight in zip(sorted_node_names, sorted_normalized_weights):
        print(f"{name}: {weight:.4f}")


def explain_single_sample(model, dataset, device, index=None, cif_name=None):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_pool)
    data_list = list(loader)

    if cif_name is not None:
        cif_names = [id_prop[0] for id_prop in dataset.id_prop_data]
        if cif_name not in cif_names:
            raise ValueError(f"Specified CIF file name {cif_name} not found in dataset")
        index = cif_names.index(cif_name)
    elif index is None:
        index = 0

    data = data_list[index]
    input_var = (
        Variable(data[0][0].to(device)),
        Variable(data[0][1].to(device)),
        data[0][2],
        data[0][3]
    )

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=config['explain_epochs']),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type=None,
        model_config=dict(
            mode='regression',
            task_level='node',
            return_type='raw',
        )
    )

    explanation = explainer(input_var[0], input_var[1], index=None,
                            nbr_fea_idx=input_var[2], crystal_atom_idx=input_var[3])

    node_mask = explanation.node_mask.cpu().detach().numpy()
    node_importance = np.sum(node_mask, axis=1)

    cif_id, _ = dataset.id_prop_data[index]

    cif_path = os.path.join(dataset.root_dir, 'cif', f"{cif_id}.cif")
    structure = Structure.from_file(cif_path)

    # 修改点：只显示原子符号和序号
    node_names = [
        f"{site.specie.symbol}{i}"
        for i, site in enumerate(structure)
    ]

    visualize_atom_importance(node_importance, node_names, cif_id)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() and not config.get('disable_cuda', False) else "cpu")
    print(f"Using device: {device}")

    model, normalizer = load_model(config['modelpath'], device)

    predict_dataset = CIFData(config['cifpath'])
    print(f"Number of samples: {len(predict_dataset)}")

    if config.get('explain_enabled', False):
        if config.get('explain_global', False):
            run_gnn_explainer_and_visualize(model, predict_dataset, device)

        if config.get('explain_cif_name', None):
            cif_name = config['explain_cif_name']
            print(f"\nExplaining material: {cif_name}")
            try:
                explain_single_sample(model, predict_dataset, device, cif_name=cif_name)
            except ValueError as e:
                print(f"Error: {str(e)}")
                cif_names = [id_prop[0] for id_prop in predict_dataset.id_prop_data]
                print(f"Available material IDs: {cif_names}")
    else:
        print("Explanation is disabled")


if __name__ == '__main__':
    main()
