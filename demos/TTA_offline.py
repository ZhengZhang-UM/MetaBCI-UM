'''
    Cross session Test on SEED dataset - based on pure EEG
    Feature extractor: EmoAdapt (self-supervised model)

'''
import os
import torch
import collections
import numpy as np
from metabci.brainda.datasets.seed import SEED
from metabci.brainda.paradigms.emotion import Emotion
from torch.utils.data import DataLoader
from metabci.brainda.algorithms.self_supervised_learning.Base import TorchDataset
import argparse
from metabci.brainda.algorithms.self_supervised_learning import EmoAdapt
from metabci.brainda.algorithms.self_supervised_learning.utils import plot_embedding
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# 导入外部通用TTA函数
from metabci.brainda.algorithms.TTA.generic_tta import generic_tta_update_mlp

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def relocate_data(X, Y, meta):
    meta_sorted = meta.copy()
    meta_sorted['subject'] = pd.to_numeric(meta_sorted['subject'])
    meta_sorted['session_num'] = meta_sorted['session'].str.extract('(\d+)').astype(int)
    meta_sorted['run_num'] = meta_sorted['run'].str.extract('(\d+)').astype(int)
    meta_sorted = meta_sorted.sort_values(['subject', 'session_num', 'run_num', 'trial_id'])
    sort_indices = meta_sorted.index
    X_sorted = X[sort_indices]
    Y_sorted = Y[sort_indices]
    meta_sorted = meta_sorted.drop(columns=['session_num', 'run_num'])
    return X_sorted, Y_sorted, meta_sorted


def get_args(file_name):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_fold', default=0, type=int)
    parser.add_argument('--ckpt_path', default=os.path.join('..', 'models', file_name), type=str)
    return parser.parse_args()


if __name__ == "__main__":
    # 1. 数据集加载
    dataset_path = r"E:\SEED"
    train_dataset = SEED(path=dataset_path, win_duration=5, sessions=[0])
    paradigm = Emotion(srate=200, channels=["FP1", "C5", "CP3", "P4"])
    X_train, Y_train, _ = paradigm.get_data(
        train_dataset, subjects=[i+1 for i in range(15)],
        return_concat=True, n_jobs=5, verbose=False
    )

    test_dataset = SEED(path=dataset_path, win_duration=5, sessions=[1])
    X_test, Y_test, _ = paradigm.get_data(
        test_dataset, subjects=[i+1 for i in range(15)],
        return_concat=True, n_jobs=5, verbose=False
    )

    # 2. 加载EmoAdapt预训练模型提取隐特征
    args = get_args(file_name='EmoAdapt')
    model_path = os.path.join(args.ckpt_path, str(args.n_fold), 'model', 'best_model_4ch.pth')
    model = torch.load(model_path, map_location='cpu', weights_only=False)['model'].to(device)

    print("===== 提取EmoAdapt latent特征 =====")
    train_x, train_y = torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.long)
    train_dataloader = DataLoader(TorchDataset(train_x, train_y), batch_size=64, shuffle=True, drop_last=False)
    test_x, test_y = torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.long)
    test_dataloader = DataLoader(TorchDataset(test_x, test_y), batch_size=64, shuffle=True, drop_last=False)

    (latent_train, train_y), (latent_test, test_y) = model.get_latent(train_dataloader), model.get_latent(test_dataloader)

    # 3. TSNE可视化
    print("\n===== 绘制TSNE嵌入图 =====")
    tsne = TSNE(n_components=2, random_state=0, init='pca', perplexity=40)
    plot_embedding(tsne.fit_transform(latent_train), train_y, "Session-0 Train t-SNE")
    tsne = TSNE(n_components=2, random_state=0, init='pca', perplexity=40)
    plot_embedding(tsne.fit_transform(latent_test), test_y, "Session-1 Test t-SNE")

    # 4. 离线MLP基准模型（无TTA）
    print("\n===== 离线基准MLP训练 =====")
    mlp_baseline = MLPClassifier(
        hidden_layer_sizes=(256, 128), activation="relu", solver="adam",
        max_iter=1000, random_state=42, early_stopping=True
    )
    mlp_baseline.fit(latent_train, train_y)
    pred_baseline = mlp_baseline.predict(latent_test)
    acc_baseline = np.mean(pred_baseline == test_y)
    print(f"离线基准MLP准确率: {acc_baseline:.4f}")

    # 5. TTA在线自适应流程（仅调用外部通用函数）
    print("\n===== 启动TTA在线增量自适应 =====")
    BUFFER_MAX = 600
    MIN_CLASSES = 2
    # 初始化TTA所需组件
    feat_buf = collections.deque(maxlen=BUFFER_MAX)
    label_buf = collections.deque(maxlen=BUFFER_MAX)
    scaler_tta = StandardScaler()
    mlp_tta = MLPClassifier(
        hidden_layer_sizes=(256, 128), activation="relu", solver="adam",
        max_iter=500, random_state=42, early_stopping=True, warm_start=True
    )
    # 初始填充训练集样本
    for ft, lab in zip(latent_train, train_y):
        feat_buf.append(ft)
        label_buf.append(lab)
    X_init = np.array(list(feat_buf))
    scaler_tta.fit(X_init)
    mlp_tta.fit(scaler_tta.transform(X_init), np.array(list(label_buf)))

    acc_record = []
    # 流式遍历测试样本，调用通用TTA函数更新
    for idx, (feat, label) in enumerate(zip(latent_test, test_y)):
        # 调用外部封装好的TTA-MLP函数，业务层无训练逻辑
        update_flag = generic_tta_update_mlp(
            feat_buffer=feat_buf,
            label_buffer=label_buf,
            scaler=scaler_tta,
            clf=mlp_tta,
            new_feat=feat,
            true_label=int(label),
            min_classes=MIN_CLASSES,
            verbose=True if idx % 50 == 0 else False
        )
        # 每10个样本评估一次精度
        if update_flag and idx % 10 == 0:
            test_std = scaler_tta.transform(latent_test)
            pred_tta = mlp_tta.predict(test_std)
            current_acc = np.mean(pred_tta == test_y)
            acc_record.append(current_acc)

    # 最终对比结果
    print("\n===== 最终对比结果 =====")
    final_std = scaler_tta.transform(latent_test)
    final_pred = mlp_tta.predict(final_std)
    acc_tta_final = np.mean(final_pred == test_y)
    print(f"离线基准MLP Acc: {acc_baseline:.4f}")
    print(f"TTA Acc: {acc_tta_final:.4f}")

    # 绘制精度变化曲线
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(acc_record)) * 10, acc_record, label="TTA-MLP Accuracy")
    plt.axhline(y=acc_baseline, color="r", linestyle="--", label="Baseline MLP")
    plt.xlabel("Processed Test Samples")
    plt.ylabel("Test Accuracy")
    plt.title("TTA Online Adaptation Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.show()