import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
def load_timeseries(dict_data, chem_site, chem_length):
    """Load data from time-series inputs"""
    data_list = []
    for path in dict_data.values():
        loaded_data = pd.read_csv(path, delimiter=",").to_numpy()
        reshaped_data = np.reshape(np.ravel(loaded_data.T), (chem_site, chem_length, 1))
        data_list.append(reshaped_data)
    return np.concatenate(data_list, axis=2)

def load_attribute(dict_data):
    """Load data from constant attributes"""
    data_list = [np.loadtxt(path, delimiter=",", skiprows=1) for path in dict_data.values()]
    return np.concatenate(data_list, axis=1)


def create_sliding_window(data, window_size, step=1):
    """
    输入: data [N, T, F]
    输出: samples [S, N, H, F]
    """
    N, T, F = data.shape
    samples = []

    # 开始滑窗
    for t in range(0, T - window_size + 1, step):
        # 截取 [N, H, F]
        window = data[:, t: t + window_size, :]
        samples.append(window)

    return np.array(samples)  # [S, N, H, F]


def filter_empty_samples(X, Y):
    """
    输入:
        X: [S, N, T, F_in]
        Y: [S, N, T, F_out]
    输出:
        X_filtered, Y_filtered
    """
    # 1. 判断哪些位置是 NaN
    # np.isnan(Y) 返回一个布尔矩阵，True表示是NaN
    # .all(axis=(1, 2, 3)) 表示：如果在 (N, T, F) 三个维度上全是 True，则该样本为 True
    is_all_missing = np.isnan(Y).all(axis=(1, 2, 3))

    # 2. 找到需要保留的样本索引 (对 is_all_missing 取反)
    keep_indices = np.where(~is_all_missing)[0]

    # 或者找到需要删除的索引（如果你只想看哪些被删了）
    drop_indices = np.where(is_all_missing)[0]
    print(f"检测到 {len(drop_indices)} 个全空样本需要删除。")

    # 3. 进行筛选
    X_filtered = X[keep_indices]
    Y_filtered = Y[keep_indices]

    print(f"筛选后形状: X {X_filtered.shape}, Y {Y_filtered.shape}")
    return X_filtered, Y_filtered

def preprocess_dynamic_data(data, train_end, log_indices=None):
    """
    处理动态数据 (X, Y)
    1. Log变换 (全量)
    2. 切分 Train/Val
    3. 标准化 (只Fit Train)
    """
    data_processed = data.copy()

    # 1. Log Transform (针对流量、降雨等长尾分布)
    if log_indices is not None:
        for idx in log_indices:
            # 加上 epsilon 防止 log(0)
            data_processed[:, :, idx] = np.log1p(data_processed[:, :, idx])

    # 2. Split
    train_data = data_processed[:, :train_end, :]
    val_data = data_processed[:, train_end:, :]

    # 3. Fit Standard Scaler (Global: across sites and time)
    # 计算 Mean/Std: 形状为 [1, 1, F]
    mean = np.nanmean(train_data, axis=(0, 1), keepdims=True)
    std = np.nanstd(train_data, axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0  # 避免除零



    train_norm = (train_data - mean) / std
    val_norm = (val_data - mean) / std

    # === 异常值截断 ===
    # 将所有超过 10 倍标准差的值，强行拉回 10
    # 阈值 10.0 可以根据实际情况调整，比如 5.0 或 8.0
    # train_norm = np.clip(train_norm, -10.0, 10.0)
    # val_norm = np.clip(val_norm, -10.0, 10.0)

    return train_norm, val_norm, mean, std


def preprocess_static_data(data, num_time_steps, log_indices=None):

    data_processed = data.copy()

    if log_indices is not None:
        for idx in log_indices:
            data_processed[:, idx] = np.log1p(data_processed[:, idx])

    # 2. Fit Standard Scaler (Global: across sites)
    # [1, F]
    mean = np.nanmean(data_processed, axis=0, keepdims=True)
    std = np.nanstd(data_processed, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    # 3. Transform
    data_norm = (data_processed - mean) / std
    data_norm = np.nan_to_num(data_norm, nan=0.0)

    # 4. Expand & Repeat [N, F] -> [N, T, F]
    c_tensor = torch.from_numpy(data_norm).float()  # [N, F]
    c_expanded = c_tensor.unsqueeze(1)  # [N, 1, F]
    c_long = c_expanded.repeat(1, num_time_steps, 1)  # [N, T, F]

    return c_long.numpy()


# 1. 准备数据转换函数
def prepare_dataloader(X, Y,batch_size=32, shuffle=True):
    # 转为 Tensor
    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y)

    # 生成 Mask
    # Mask = 1 表示有数据，Mask = 0 表示缺失
    Mask_tensor = ~torch.isnan(Y_tensor)  # bool 类型
    Mask_tensor = Mask_tensor.float()  # 转为 float (0.0, 1.0)

    # 既然已经有了 Mask，我们需要把 Y 里的 NaN 替换成 0，防止计算 Loss 时报错
    # (虽然 Loss 会乘 Mask 过滤掉，但输入不能有 NaN)
    Y_tensor = torch.nan_to_num(Y_tensor, nan=0.0)

    # 3. 封装 Dataset
    # 我们把 X, Y, Mask 三个一起放进去
    dataset = TensorDataset(X_tensor, Y_tensor,Mask_tensor)

    # 4. 创建 DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader