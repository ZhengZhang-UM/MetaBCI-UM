# generic_tta.py
import collections
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def generic_tta_update(
    feat_buffer: collections.deque,
    label_buffer: collections.deque,
    scaler: StandardScaler,
    clf: SVC,
    new_feat: np.ndarray,
    true_label: int,
    min_classes: int = 2,
    verbose: bool = True
) -> bool:
    """
    通用TTA在线增量更新：SVM版本
    :param feat_buffer: 特征循环队列 deque(maxlen=xxx)
    :param label_buffer: 标签循环队列 deque(maxlen=xxx)
    :param scaler: StandardScaler 标准化器实例
    :param clf: SVC 分类器实例
    :param new_feat: 单条一维特征向量 (n_feat,)
    :param true_label: 数字形式真实标签 int
    :param min_classes: 训练所需最少类别数，不足则不更新
    :param verbose: 是否打印日志
    :return: bool True=模型完成更新 False=未更新
    """
    feat_buffer.append(new_feat)
    label_buffer.append(true_label)

    if verbose:
        print(f"\n📝 TTA-SVM新增标注样本 | 标签：{true_label}，总样本数：{len(feat_buffer)}")

    unique_labels = np.unique(list(label_buffer))
    if len(unique_labels) < min_classes:
        if verbose:
            print(f"⚠️ 跳过更新：仅 {len(unique_labels)} 类，至少需要{min_classes}类")
        return False

    X_train = np.array(list(feat_buffer))
    y_train = np.array(list(label_buffer))
    scaler.fit(X_train)
    X_std = scaler.transform(X_train)
    clf.fit(X_std, y_train)

    if verbose:
        print("🔄 TTA-SVM模型更新完成")
    return True


def generic_tta_update_mlp(
    feat_buffer: collections.deque,
    label_buffer: collections.deque,
    scaler: StandardScaler,
    clf: MLPClassifier,
    new_feat: np.ndarray,
    true_label: int,
    min_classes: int = 2,
    verbose: bool = True
) -> bool:
    """
    通用TTA在线增量更新：MLP版本
    :param feat_buffer: 特征循环队列 deque(maxlen=xxx)
    :param label_buffer: 标签循环队列 deque(maxlen=xxx)
    :param scaler: StandardScaler 标准化器实例
    :param clf: MLPClassifier 分类器实例（建议开启warm_start=True）
    :param new_feat: 单条一维特征向量 (n_feat,)
    :param true_label: 数字形式真实标签 int
    :param min_classes: 训练所需最少类别数，不足则不更新
    :param verbose: 是否打印日志
    :return: bool True=模型完成更新 False=未更新
    """
    feat_buffer.append(new_feat)
    label_buffer.append(true_label)

    if verbose:
        print(f"\n📝 TTA-MLP新增标注样本 | 标签：{true_label}，总样本数：{len(feat_buffer)}")

    unique_labels = np.unique(list(label_buffer))
    if len(unique_labels) < min_classes:
        if verbose:
            print(f"⚠️ 跳过更新：仅 {len(unique_labels)} 类，至少需要{min_classes}类")
        return False

    X_train = np.array(list(feat_buffer))
    y_train = np.array(list(label_buffer))
    scaler.fit(X_train)
    X_std = scaler.transform(X_train)
    clf.fit(X_std, y_train)

    if verbose:
        print("🔄 TTA-MLP模型更新完成")
    return True