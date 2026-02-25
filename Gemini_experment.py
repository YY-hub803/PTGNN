import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import copy
import random
import sys

# ==========================================
# 0. 全局配置与超参数 (Configuration)
# ==========================================
CONFIG = {
    'seq_length': 30,
    'pred_step': 0,
    'hidden_dim': 128,  # 增大隐藏层容量 (64 -> 128)
    'batch_size': 32,
    'pretrain_epochs': 80,  # 预训练不用太多轮
    'finetune_epochs': 200,
    'lr_pre': 0.005,
    'lr_fine': 0.002,  # 微调初始学习率稍大，配合Scheduler
    'dropout': 0.4,  # 增加Dropout防止过拟合
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_dir': './',
    'patience': 25
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(CONFIG['seed'])


# ==========================================
# 1. 精细化数据工程 (Robust Data Engineering)
# ==========================================
class RiverBasinDataset(Dataset):
    def __init__(self, config, mode='train', target_type='TP', scalers=None):
        self.config = config
        self.mode = mode
        self.target_type = target_type
        self.scalers = scalers if scalers else {}
        self.load_and_process()

    def load_and_process(self):
        dir_ = self.config['data_dir']

        # --- A. 基础节点信息 ---
        try:
            tp_df = pd.read_csv(f'{dir_}input_yobs_TP.csv')
        except FileNotFoundError:
            print(f"错误：找不到文件 {dir_}input_yobs_TP.csv")
            sys.exit(1)

        self.nodes = tp_df.columns.tolist()
        self.nodes = [n for n in self.nodes if 'Unnamed' not in n and 'Date' not in n]
        self.num_nodes = len(self.nodes)

        if self.mode == 'train' and self.target_type == 'Flow':
            print(f"检测到 {self.num_nodes} 个子流域节点。")

        # --- B. 静态属性 ---
        static_df = pd.read_csv(f'{dir_}input_c_all.csv')
        hydro_candidates = ['Impervious', 'Forest', 'Grassland', 'Wate', 'Wetland', 'GRAV__mean', 'POR__mean', 'Slope']
        pollut_candidates = ['Cropland', 'Pop', 'Light', 'P_gdp', 'TN__mean', 'TP__mean', 'AP__mean']

        def get_valid_features(candidates, df):
            valid_cols = [c for c in candidates if c in df.columns]
            data = df[valid_cols].iloc[:self.num_nodes].fillna(0).values
            return data

        raw_hydro = get_valid_features(hydro_candidates, static_df)
        raw_pollut = get_valid_features(pollut_candidates, static_df)

        self.x_stat_hydro = torch.FloatTensor(MinMaxScaler().fit_transform(raw_hydro)).to(CONFIG['device'])
        self.x_stat_pollut = torch.FloatTensor(MinMaxScaler().fit_transform(raw_pollut)).to(CONFIG['device'])

        # --- C. 动态驱动 ---
        flow_df = pd.read_csv(f'{dir_}input_xforce_Discharge.csv')
        self.flow_data = np.nan_to_num(flow_df[self.nodes].values)

        meteo_files = ['input_xforce_pre_sum.csv', 'input_xforce_A_t.csv', 'input_xforce_shum.csv']
        meteo_list = []
        for f in meteo_files:
            if os.path.exists(f'{dir_}{f}'):
                df = pd.read_csv(f'{dir_}{f}')
                data = np.nan_to_num(df[self.nodes].values)
                meteo_list.append(data)

        if not meteo_list: sys.exit(1)
        self.x_dynamic = np.stack(meteo_list, axis=2)

        # 归一化 (使用传入的scaler或新建)
        T, N, F = self.x_dynamic.shape
        if 'dynamic' not in self.scalers:
            self.scalers['dynamic'] = StandardScaler()
            self.x_dynamic = self.scalers['dynamic'].fit_transform(self.x_dynamic.reshape(-1, F)).reshape(T, N, F)
        else:
            self.x_dynamic = self.scalers['dynamic'].transform(self.x_dynamic.reshape(-1, F)).reshape(T, N, F)

        if 'flow' not in self.scalers:
            self.scalers['flow'] = StandardScaler()
            self.flow_normalized = self.scalers['flow'].fit_transform(self.flow_data.reshape(-1, 1)).reshape(T, N)
        else:
            self.flow_normalized = self.scalers['flow'].transform(self.flow_data.reshape(-1, 1)).reshape(T, N)

        # --- D. 图结构 ---
        edge_df = pd.read_csv(f'{dir_}edge_info.csv')
        u, v = edge_df['source'].values, edge_df['target'].values
        if max(u.max(), v.max()) >= self.num_nodes:
            print("错误：图结构索引越界，请检查CSV。")
            sys.exit(1)

        # 修复 UserWarning: explicitly converting to numpy array first
        self.edge_index = torch.tensor(np.array([u, v]), dtype=torch.long).to(CONFIG['device'])

        # --- E. 目标变量 (关键修改: TP 标准化) ---
        self.tp_raw = tp_df[self.nodes].values

        # 仅对非NaN值进行统计以构建Scaler
        if 'tp' not in self.scalers:
            valid_tp = self.tp_raw[~np.isnan(self.tp_raw)].reshape(-1, 1)
            self.scalers['tp'] = StandardScaler()
            self.scalers['tp'].fit(valid_tp)

        # 预处理 TP 数据 (保持NaN位置)
        T_tp, N_tp = self.tp_raw.shape
        tp_flat = self.tp_raw.reshape(-1, 1)
        valid_mask = ~np.isnan(tp_flat)

        tp_scaled_flat = np.full_like(tp_flat, np.nan)

        # [Fix: ValueError] 确保输入 transform 的是 2D 数组 (-1, 1)
        if np.any(valid_mask):
            valid_data = tp_flat[valid_mask].reshape(-1, 1)  # Force 2D
            tp_scaled_flat[valid_mask] = self.scalers['tp'].transform(valid_data).flatten()

        self.tp_data = tp_scaled_flat.reshape(T_tp, N_tp)

        self.generate_samples()

    def generate_samples(self):
        self.samples = []
        seq_len = self.config['seq_length']
        total_time = self.x_dynamic.shape[0]
        split_idx = int(total_time * 0.8)

        range_t = range(seq_len, split_idx) if self.mode == 'train' else range(split_idx, total_time)

        for t in range_t:
            x_meteo = self.x_dynamic[t - seq_len:t]  # [30, N, 3]

            if self.target_type == 'Flow':
                # 【预训练】：虽然不看流量，但为了维度对齐，造一个全0的流量特征
                # 这样模型结构始终是 Input=4
                x_flow_dummy = np.zeros_like(x_meteo[:, :, 0:1])  # [30, N, 1] 全0
                x_input = np.concatenate([x_meteo, x_flow_dummy], axis=2)

                y_target = self.flow_normalized[t, :]
                mask = np.ones_like(y_target, dtype=bool)

            else:
                # 【微调 TP】：放入真实的流量数据
                # 这里使用的是 t-30 到 t-1 的流量 (为了严谨，不含当天，模拟预测场景)
                x_flow_real = self.flow_normalized[t - seq_len:t][:, :, np.newaxis]
                x_input = np.concatenate([x_meteo, x_flow_real], axis=2)

                y_target = self.tp_data[t, :]
                mask = ~np.isnan(y_target)

            if self.target_type == 'TP' and np.sum(mask) == 0: continue
            self.samples.append((x_input, np.nan_to_num(y_target), mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, mask = self.samples[idx]
        return (torch.FloatTensor(x).to(CONFIG['device']),
                torch.FloatTensor(y).to(CONFIG['device']),
                torch.BoolTensor(mask).to(CONFIG['device']))


# ==========================================
# 2. 物理引导图神经网络 (Physics-Guided Model)
# ==========================================
class PhysicsGatedLSTM(nn.Module):
    def __init__(self, dyn_dim, hydro_stat_dim, pollut_stat_dim, hidden_dim):
        super().__init__()
        self.hydro_emb = nn.Linear(hydro_stat_dim, 16)
        self.pollut_emb = nn.Linear(pollut_stat_dim, 16)

        self.input_gate = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, dyn_dim), nn.Sigmoid()
        )
        # 输入: Dyn(4) + Stat(32) = 36
        self.lstm = nn.LSTM(dyn_dim + 32, hidden_dim, batch_first=True, num_layers=2, dropout=CONFIG['dropout'])

    def forward(self, x_dyn, x_hydro, x_pollut):
        B, S, N, F = x_dyn.size()
        h_emb = self.hydro_emb(x_hydro)
        p_emb = self.pollut_emb(x_pollut)
        stat_combined = torch.cat([h_emb, p_emb], dim=-1)  # [N, 32]

        stat_for_gate = stat_combined.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        gate_values = self.input_gate(stat_for_gate)
        x_dyn_gated = x_dyn * gate_values

        stat_expanded = stat_combined.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
        lstm_input = torch.cat([x_dyn_gated, stat_expanded], dim=-1)

        lstm_input_flat = lstm_input.view(B * N, S, -1)
        out, _ = self.lstm(lstm_input_flat)
        h_t = out[:, -1, :].view(B, N, -1)
        return h_t


class PG_TL_GNN(nn.Module):
    def __init__(self, num_nodes, dyn_dim, hydro_dim, pollut_dim, hidden_dim):
        super().__init__()
        self.backbone = PhysicsGatedLSTM(dyn_dim, hydro_dim, pollut_dim, hidden_dim)
        self.gnn = GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False, dropout=CONFIG['dropout'])

        self.head_flow = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.head_tp = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Dropout(CONFIG['dropout']),
            nn.Linear(64, 1)
        )

    def forward(self, x_dyn, x_hydro, x_pollut, edge_index, task='TP'):
        h_temp = self.backbone(x_dyn, x_hydro, x_pollut)

        # 简单的批处理 GNN
        B, N, H = h_temp.size()
        h_spatial_list = []
        for i in range(B):
            h_spatial_list.append(self.gnn(h_temp[i], edge_index))
        h_spatial = torch.stack(h_spatial_list, dim=0)

        h_final = h_temp + h_spatial  # Residual

        if task == 'Flow':
            out = self.head_flow(h_final)
        else:
            out = self.head_tp(h_final)
        return out.squeeze(-1)


# ==========================================
# 3. 实验流程 (Experiment Workflow)
# ==========================================
def calc_nse(obs, sim):
    """计算 NSE"""
    obs_var = np.var(obs)
    if obs_var < 1e-6: return 0.0
    mse = np.mean((obs - sim) ** 2)
    return 1 - (mse / obs_var)


def run_experiment():
    # 1. 初始化训练集 (Flow)
    ds_pre_train = RiverBasinDataset(CONFIG, mode='train', target_type='Flow')
    dl_pre_train = DataLoader(ds_pre_train, batch_size=CONFIG['batch_size'], shuffle=True)

    # 获取共享 Scalers
    shared_scalers = ds_pre_train.scalers

    # 2. 初始化微调集 (TP) - 必须使用相同的 Scaler!
    ds_fine_train = RiverBasinDataset(CONFIG, mode='train', target_type='TP', scalers=shared_scalers)
    ds_fine_test = RiverBasinDataset(CONFIG, mode='test', target_type='TP', scalers=shared_scalers)
    dl_fine_train = DataLoader(ds_fine_train, batch_size=CONFIG['batch_size'], shuffle=True)
    dl_fine_test = DataLoader(ds_fine_test, batch_size=CONFIG['batch_size'], shuffle=False)

    sample_x, _, _ = ds_pre_train[0]
    dims = (sample_x.shape[2], ds_pre_train.x_stat_hydro.shape[1], ds_pre_train.x_stat_pollut.shape[1])

    model = PG_TL_GNN(ds_pre_train.num_nodes, *dims, CONFIG['hidden_dim']).to(CONFIG['device'])

    # --- Phase 1: Pre-training (Flow) ---
    print("\n>>> Phase 1: Pre-training (Flow)...")
    opt_pre = torch.optim.Adam(model.parameters(), lr=CONFIG['lr_pre'])
    loss_fn = nn.SmoothL1Loss()  # Huber Loss

    for epoch in range(CONFIG['pretrain_epochs']):
        model.train()
        total_loss = 0
        for x, y, _ in dl_pre_train:
            opt_pre.zero_grad()
            pred = model(x, ds_pre_train.x_stat_hydro, ds_pre_train.x_stat_pollut, ds_pre_train.edge_index, task='Flow')
            loss = loss_fn(pred, y)
            loss.backward()
            opt_pre.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  [Pre] Ep {epoch + 1} | Loss: {total_loss / len(dl_pre_train):.4f}")

    # --- Phase 2: Fine-tuning (TP) ---
    print("\n>>> Phase 2: Fine-tuning (TP)...")

    # 关键策略: 差分学习率
    # Backbone 学习率低 (0.1x)，Head 学习率高 (1.0x)
    fine_params = [
        {'params': model.backbone.parameters(), 'lr': CONFIG['lr_fine'] * 0.1},
        {'params': model.gnn.parameters(), 'lr': CONFIG['lr_fine'] * 0.1},
        {'params': model.head_tp.parameters(), 'lr': CONFIG['lr_fine']}
    ]
    opt_fine = torch.optim.Adam(fine_params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_fine, mode='max', factor=0.5, patience=10, verbose=True)

    best_nse = -999
    patience_cnt = 0
    tp_scaler = shared_scalers['tp']

    for epoch in range(CONFIG['finetune_epochs']):
        model.train()
        train_loss = 0

        for x, y, mask in dl_fine_train:
            opt_fine.zero_grad()
            pred = model(x, ds_fine_train.x_stat_hydro, ds_fine_train.x_stat_pollut, ds_fine_train.edge_index,
                         task='TP')

            # Masked Loss on Normalized Data
            loss = F.smooth_l1_loss(pred[mask], y[mask])
            loss.backward()
            opt_fine.step()
            train_loss += loss.item()

        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            preds, trues, masks = [], [], []
            with torch.no_grad():
                for x, y, mask in dl_fine_test:
                    p = model(x, ds_fine_train.x_stat_hydro, ds_fine_train.x_stat_pollut, ds_fine_train.edge_index,
                              task='TP')
                    preds.append(p.cpu().numpy())
                    trues.append(y.cpu().numpy())
                    masks.append(mask.cpu().numpy())

            # 反归一化 (Inverse Transform) 以计算真实的 NSE
            p_norm = np.concatenate(preds)
            t_norm = np.concatenate(trues)
            m = np.concatenate(masks)

            # 只在有效点计算
            valid_p_norm = p_norm[m]
            valid_t_norm = t_norm[m]

            # 还原到原始物理量纲 (mg/L)
            valid_p_raw = tp_scaler.inverse_transform(valid_p_norm.reshape(-1, 1)).flatten()
            valid_t_raw = tp_scaler.inverse_transform(valid_t_norm.reshape(-1, 1)).flatten()

            if len(valid_t_raw) > 0:
                nse = calc_nse(valid_t_raw, valid_p_raw)
                scheduler.step(nse)  # 根据 NSE 调整学习率

                print(f"  [Fine] Ep {epoch + 1} | Loss: {train_loss / len(dl_fine_train):.4f} | Val NSE: {nse:.4f}")

                if nse > best_nse:
                    best_nse = nse
                    torch.save(model.state_dict(), 'best_model_tp.pth')
                    patience_cnt = 0
                    print("    -> New Best Model Saved!")
                else:
                    patience_cnt += 1

            if patience_cnt >= CONFIG['patience']:
                print("Early Stopping.")
                break

    print(f"\nFinal Best NSE: {best_nse:.4f}")


if __name__ == "__main__":
    run_experiment()