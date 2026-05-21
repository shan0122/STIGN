import torch                                     # 导入 PyTorch 核心库
from torch.utils.data import Dataset, DataLoader # 导入 PyTorch 数据处理的基类 Dataset 和批量加载器 DataLoader
import numpy as np                               # 导入 NumPy，用于读取和处理硬盘上的 .npy 数组文件
import os                                        # 导入 os 模块，用于处理文件路径
import json                                      # 导入 json 模块，用于解析 JSON 标注文件

# ==========================================
# 自定义数据集类，必须继承自 torch.utils.data.Dataset
# ==========================================
class MSVDDataset(Dataset):
    # 初始化函数，在创建数据集实例时被调用
    def __init__(self, feature_dir, annotation_file, max_seq_len=26, num_semantic_concepts=300):
        super().__init__()                       # 调用父类的初始化函数
        self.feature_dir = feature_dir           # 保存特征文件（.npy）所在的目录路径
        self.max_seq_len = max_seq_len           # 设定句子（字幕）的最大长度，用于对齐张量形状
        self.samples = []                        # 初始化一个空列表，用于存放所有的训练样本信息
        
        # 检查指定的 JSON 标注文件是否存在
        if os.path.exists(annotation_file):
            # 打开 JSON 文件
            with open(annotation_file, 'r') as f:
                data = json.load(f)              # 将 JSON 文件内容加载为 Python 列表/字典
                
                # 遍历 JSON 中的每一个视频信息
                for item in data:
                    vid = item['video_id']               # 获取当前视频的 ID
                    semantic_labels = item['semantic_labels'] # 获取该视频的全局语义属性标签 (例如一个 300 维的 multi-hot 向量)
                    
                    # 关键展开操作：一个视频通常对应多个人类写的句子 (tokens_list)
                    # 遍历这个视频的所有句子，把它们展开成独立的样本
                    for token_seq in item['tokens_list']:
                        # 将每个“视频-句子”对作为一个独立的样本存入列表中
                        self.samples.append({
                            'video_id': vid,
                            'tokens': token_seq,         # 当前这句字幕的单词索引列表
                            'semantic_labels': semantic_labels # 视频的语义标签（所有句子共享相同的视觉标签）
                        })

    # 魔法方法 __len__：返回数据集的总样本数，DataLoader 需要这个来计算总共有多少个 batch
    def __len__(self):
        return len(self.samples)

    # 魔法方法 __getitem__：根据给定的索引 idx，返回单个样本的所有数据
    # 这是 DataLoader 每次取数据时调用的核心函数
    def __getitem__(self, idx):
        # 1. 从列表中取出对应的样本字典
        sample = self.samples[idx]
        vid_id = sample['video_id']
        
        # 2. 读取视觉特征文件
        try:
            # 读取局部目标特征，形状通常为 (16帧, 5个目标, 512维)
            roi_feats = np.load(os.path.join(self.feature_dir, f"{vid_id}_roi.npy"))
            # 读取目标的边界框坐标，形状为 (16帧, 5个目标, 4个坐标值[x1,y1,x2,y2])
            coords = np.load(os.path.join(self.feature_dir, f"{vid_id}_coords.npy"))
            # 读取全局视频特征 (例如 R3D 输出)，形状通常为 (1024维,)
            global_feats = np.load(os.path.join(self.feature_dir, f"{vid_id}_global.npy"))
        except:
            # 异常处理：如果在硬盘上找不到这个视频的 .npy 文件，或者文件损坏
            # 为了防止程序崩溃导致整个训练中断，返回与正常特征形状一致的全 0 占位符
            roi_feats = np.zeros((16, 5, 512), dtype=np.float32)
            coords = np.zeros((16, 5, 4), dtype=np.float32)
            global_feats = np.zeros((1024,), dtype=np.float32)
        
        # 3. 处理语言序列 (Padding / Truncation)
        # 截断：如果句子实际长度超过了最大长度 max_seq_len，则截去多余的部分
        tokens = sample['tokens'][:self.max_seq_len]
        # 填充：如果句子实际长度不足 max_seq_len，在末尾补充 0（0 通常代表 <PAD> 标签）
        # 这样能保证在一个 Batch 里，所有句子的长度是一致的，才能组合成 Tensor
        padded_tokens = tokens + [0] * (self.max_seq_len - len(tokens)) 
        
        # 4. 将所有的 NumPy 数组和 Python 列表转换为 PyTorch 的 Tensor，并打包成字典返回
        return {
            'video_id': vid_id,                                              # 视频 ID (字符串，不参与梯度计算，用于推理时跟踪)
            'roi_feats': torch.tensor(roi_feats, dtype=torch.float32),       # 转换为 32 位浮点张量
            'coords': torch.tensor(coords, dtype=torch.float32),             # 转换为 32 位浮点张量
            'global_feats': torch.tensor(global_feats, dtype=torch.float32), # 转换为 32 位浮点张量
            'tokens': torch.tensor(padded_tokens, dtype=torch.long),         # 单词索引必须转换为 64 位整数 (long) 张量，这是 embedding 层要求的
            'semantic_labels': torch.tensor(sample['semantic_labels'], dtype=torch.float32) # 语义标签张量
        }

# ==========================================
# 辅助函数，用于快速创建不同拆分集（train, val, test）的 DataLoader
# ==========================================
def get_dataloader(config, split='train'):
    # 拼接 JSON 标注文件的完整路径，例如 "data/processed/msvd_train.json"
    # 如果是 MSVD 就是 msvd_，如果是 Teacher 就是 teacher_
    prefix = "teacher" if "Teacher" in config['data']['processed_dir'] else "msvd"
    anno_file = os.path.join(config['data']['processed_dir'], f"{prefix}_{split}.json")
    
    # 实例化上面定义的 MSVDDataset 类
    dataset = MSVDDataset(
        feature_dir=config['data']['feature_dir'], 
        annotation_file=anno_file, 
        max_seq_len=config['data']['max_seq_len'], 
        num_semantic_concepts=config['model']['num_semantic_concepts']
    )
    
    # 创建并返回 DataLoader
    # batch_size: 每次并行处理的样本数 (如 32, 64)
    # shuffle: 如果是训练集 ('train')，则每个 epoch 打乱数据顺序；如果是测试集，则不打乱
    # num_workers=0: 使用主进程进行数据加载。在 Mac 的 MPS 后端或 Windows 上，多进程加载有时会报错，设为 0 是最安全的选择。
    return DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=(split=='train'), num_workers=0)