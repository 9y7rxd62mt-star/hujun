import pandas as pd
import numpy as np
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import time
from tqdm import tqdm

# ==========================================
# 1. 读取数据，获取唯一的污染物名称
# ==========================================
file_path = r'C:\Users\15787\PycharmProjects\PythonProject\Pollutant Experimental Data.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 提取所有唯一的污染物名称（去除NaN）
pollutants = df['pollutant'].dropna().unique()
print(f"发现 {len(pollutants)} 种独特的污染物：\n{pollutants}")


# ==========================================
# 2. 通过PubChem获取SMILES（自动缓存）
# ==========================================
def get_smiles_from_name(name):
    """通过PubChem API获取SMILES，失败返回None"""
    try:
        # 先尝试直接按名称搜索化合物
        compounds = pcp.get_compounds(name, 'name')
        if compounds:
            # 返回标准SMILES（通常第一个结果最相关）
            return compounds[0].canonical_smiles
        else:
            # 尝试处理同义词或常见别名（例如“氯乙烯”会成功）
            return None
    except Exception as e:
        print(f"获取 {name} 的SMILES失败: {e}")
        return None


# 构建名称->SMILES的映射（避免重复请求）
name_to_smiles = {}
for p in tqdm(pollutants, desc="正在从PubChem获取SMILES"):
    if p not in name_to_smiles:
        smiles = get_smiles_from_name(p)
        name_to_smiles[p] = smiles
        time.sleep(0.2)  # 避免请求过快
    print(f"{p} -> {smiles}")


# ==========================================
# 3. 定义要计算的分子描述符函数
# ==========================================
def compute_descriptors(smiles):
    """输入SMILES，返回一个包含多个描述符的字典"""
    if smiles is None:
        return {desc: np.nan for desc in descriptor_names}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {desc: np.nan for desc in descriptor_names}

    # 计算常用描述符
    desc = {
        'MolWt': Descriptors.ExactMolWt(mol),  # 分子量
        'LogP': Descriptors.MolLogP(mol),  # 油水分配系数
        'NumHDonors': Lipinski.NumHDonors(mol),  # 氢键供体数
        'NumHAcceptors': Lipinski.NumHAcceptors(mol),  # 氢键受体数
        'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),  # 可旋转键数
        'TPSA': Descriptors.TPSA(mol),  # 拓扑极性表面积
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),  # 芳香环数
        'FractionCsp3': Descriptors.FractionCsp3(mol),  # sp3碳比例
        'NumHalogens': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [9, 17, 35, 53]),  # 卤原子数
    }
    return desc


descriptor_names = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors',
                    'NumRotatableBonds', 'TPSA', 'NumAromaticRings',
                    'FractionCsp3', 'NumHalogens']

# ==========================================
# 4. 为每个污染物计算描述符
# ==========================================
print("\n正在计算分子描述符...")
desc_data = []
for p in tqdm(pollutants):
    smiles = name_to_smiles[p]
    desc = compute_descriptors(smiles)
    desc['pollutant'] = p
    desc_data.append(desc)

desc_df = pd.DataFrame(desc_data)

# ==========================================
# 5. 将描述符合并回原始数据
# ==========================================
# 将描述符表按污染物名称左连接回原数据
df_enhanced = df.merge(desc_df, on='pollutant', how='left')

# ==========================================
# 6. 保存增强后的数据
# ==========================================
output_path = 'TestData_with_descriptors.xlsx'
df_enhanced.to_excel(output_path, index=False)
print(f"\n✅ 成功！已保存包含分子描述符的数据集至：{output_path}")
print(f"新增的特征列：{descriptor_names}")