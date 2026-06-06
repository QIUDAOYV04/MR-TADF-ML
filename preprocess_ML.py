import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdFingerprintGenerator
import pandas as pd

from config import DATASET_DIR, MORGAN_RADIUS, MORGAN_BITS

# ============================================================
# 1. RDKit 2D 描述符计算（约210个数值型特征）
# ============================================================
def compute_rdkit_descriptors(mol):
    """计算单个分子的全套RDKit 2D描述符，返回numpy数组"""
    descriptor_names = [desc[0] for desc in Descriptors.descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    desc_values = calculator.CalcDescriptors(mol)
    return np.array(desc_values, dtype=np.float64)
# ============================================================
# 2. Morgan指纹计算（2048位二值向量）
# ============================================================
def compute_morgan_fp(mol, radius=2, n_bits=2048):
    """计算单个分子的Morgan指纹，返回numpy数组"""
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = mfpgen.GetFingerprint(mol)
    return np.array(fp, dtype=np.float32)

# ============================================================
# 3. MACCS指纹计算（167位二值向量）
# ============================================================
def compute_maccs_fp(mol):
    """计算单个分子的MACCS指纹，返回numpy数组"""
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp, dtype=np.float32)

# ============================================================
# 4. 获取描述符名称列表（供后续特征筛选和SHAP分析使用）
# ============================================================
def get_feature_names(morgan_bits=2048):
    """返回全部特征的名称列表，顺序与特征向量拼接顺序一致"""
    # RDKit 2D描述符名称
    rdkit_names = [desc[0] for desc in Descriptors.descList]
    # Morgan指纹名称
    morgan_names = [f"MorganFP_{i}" for i in range(morgan_bits)]
    # MACCS指纹名称
    maccs_names = [f"MACCS_{i}" for i in range(167)]

    return rdkit_names + morgan_names + maccs_names


# ============================================================
# 5. 数据集划分（和原来一样）
# ============================================================
def split_dataset(X, y_est, y_krisc, ratio):
    """Shuffle and split a dataset."""
    np.random.seed(42)
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    split_point = int(ratio * n)

    train_idx = indices[:split_point]
    dev_idx = indices[split_point:]

    return (X[train_idx], y_est[train_idx], y_krisc[train_idx],
            X[dev_idx], y_est[dev_idx], y_krisc[dev_idx])


# ============================================================
# 6. 核心函数：创建数据集（替代原来的 create_datasets）
# ============================================================
def create_datasets(task, dataset, radius=2, device=None):
    """
    读取数据文件，计算分子特征，返回numpy数组。

    参数保留 task, dataset, radius, device 和原接口一致，
    其中 device 在传统ML中不使用，仅保留接口兼容性。

    返回:
        X_train, y_train_est, y_train_krisc,
        X_dev, y_dev_est, y_dev_krisc,
        X_test, y_test_est, y_test_krisc,
        feature_names
    """
    dir_dataset = DATASET_DIR / task / dataset

    def process_file(filename):
        """读取一个数据文件，返回特征矩阵和标签数组"""
        filepath = dir_dataset / filename
        print(f"正在处理: {filepath}")

        with open(filepath, 'r', encoding='utf-8-sig') as f:
            data_original = f.read().strip().replace('\r\n', '\n').replace('\r', '\n').split('\n')

        X_list = []
        est_list = []
        krisc_list = []
        skipped = 0

        for line_idx, line in enumerate(data_original):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # 安全检查：至少需要SMILES + 两个数值 = 3个部分
            if len(parts) < 3:
                print(f"  第{line_idx + 1}行格式异常（只有{len(parts)}列），跳过: {line[:80]}...")
                skipped += 1
                continue

            try:
            # 从右边取最后两个数值，前面全部当作SMILES
                krisc_val = float(parts[-1])
                est_val = float(parts[-2])
                smiles = ' '.join(parts[:-2])
            except ValueError:
                print(f"  第{line_idx+1}行数值解析失败，跳过: {line[:80]}...")
                skipped += 1
                continue
            # 解析SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"  跳过无效SMILES: {smiles[:60]}...")
                skipped += 1
                continue

            try:
                # 计算三类特征
                rdkit_desc = compute_rdkit_descriptors(mol)
                morgan_fp = compute_morgan_fp(mol, radius=radius, n_bits=MORGAN_BITS)
                maccs_fp = compute_maccs_fp(mol)

                # 拼接成一个长向量
                feature_vector = np.concatenate([rdkit_desc, morgan_fp, maccs_fp])

                X_list.append(feature_vector)
                est_list.append(est_val)
                krisc_list.append(krisc_val)

            except Exception as e:
                print(f"  计算失败: {smiles[:60]}... 错误: {e}")
                skipped += 1
                continue

        print(f"  成功: {len(X_list)} 个分子, 跳过: {skipped} 个")

        X = np.array(X_list, dtype=np.float64)
        y_est = np.array(est_list, dtype=np.float32)
        y_krisc = np.array(krisc_list, dtype=np.float32)

        return X, y_est, y_krisc

    # 处理训练集和测试集
    X_train_all, y_train_est_all, y_train_krisc_all = process_file('data_train.txt')
    X_test, y_test_est, y_test_krisc = process_file('data_test.txt')

    # 处理NaN和Inf：替换为0（后续特征筛选会处理这些列）
    X_train_all = np.nan_to_num(X_train_all, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    # 截断
    f32_max = np.finfo(np.float32).max
    f32_min = np.finfo(np.float32).min
    X_train_all = np.clip(X_train_all, a_min=f32_min, a_max=f32_max)
    X_test = np.clip(X_test, a_min=f32_min, a_max=f32_max)
    # 修改这里：在清理完所有的极值和 Inf 之后，再安全地降维到 float32
    X_train_all = X_train_all.astype(np.float32)
    X_test = X_test.astype(np.float32)
    # 训练集拆分为train和dev（75%:25%，和原来一致）
    (X_train, y_train_est, y_train_krisc,
     X_dev, y_dev_est, y_dev_krisc) = split_dataset(
        X_train_all, y_train_est_all, y_train_krisc_all, ratio=0.75
    )

    # 获取特征名称
    feature_names = get_feature_names(morgan_bits=MORGAN_BITS)

    print(f"\n特征维度: {X_train.shape[1]}")
    print(f"  RDKit 2D描述符: {len(Descriptors.descList)} 个")
    print(f"  Morgan指纹: 2048 位")
    print(f"  MACCS指纹: 167 位")
    print(f"  总计: {len(feature_names)} 个特征")
    print(f"\n训练集: {X_train.shape[0]} 个样本")
    print(f"验证集: {X_dev.shape[0]} 个样本")
    print(f"测试集: {X_test.shape[0]} 个样本")

    return (X_train, y_train_est, y_train_krisc,
            X_dev, y_dev_est, y_dev_krisc,
            X_test, y_test_est, y_test_krisc,
            feature_names)


# ============================================================
# 直接运行时的测试代码
# ============================================================
if __name__ == "__main__":
    from config import DEFAULT_TASK, DEFAULT_DATASET
    print("=" * 70)
    print("测试特征提取流程")
    print("=" * 70)

    results = create_datasets(
        task=DEFAULT_TASK,
        dataset=DEFAULT_DATASET,
        radius=MORGAN_RADIUS,
        device=None
    )

    (X_train, y_train_est, y_train_krisc,
     X_dev, y_dev_est, y_dev_krisc,
     X_test, y_test_est, y_test_krisc,
     feature_names) = results

    print("\n" + "=" * 70)
    print("数据形状检查")
    print("=" * 70)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_dev shape:   {X_dev.shape}")
    print(f"X_test shape:  {X_test.shape}")

    print(f"\ny_train_est 范围: [{y_train_est.min():.4f}, {y_train_est.max():.4f}]")
    print(f"y_train_krisc 范围: [{y_train_krisc.min():.4f}, {y_train_krisc.max():.4f}]")

    # 统计-999缺失值
    est_valid = (y_train_est != -999.0).sum()
    krisc_valid = (y_train_krisc != -999.0).sum()
    print(f"\n训练集ΔEST有效样本: {est_valid}/{len(y_train_est)}")
    print(f"训练集kRISC有效样本: {krisc_valid}/{len(y_train_krisc)}")

    # 打印前10个特征名称示例
    print(f"\n特征名称示例（前10个）:")
    for i, name in enumerate(feature_names[:10]):
        print(f"  [{i}] {name}")
    print(f"  ...")
    print(f"  [{len(feature_names)-1}] {feature_names[-1]}")

    print("\n特征提取完成！可以接入传统ML模型训练。")
    df = pd.DataFrame(X_train, columns=feature_names)
    df['Target_est'] = y_train_est
    df['Target_krisc'] = y_train_krisc
    excel_filename = "molecular_features_preview.xlsx"

    df.head(229).to_excel(excel_filename, index=False)
