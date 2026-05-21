import torch                                      # 导入 PyTorch 核心库
import torchvision.models as models               # 导入 torchvision 中预训练好的经典 CNN 模型
import torchvision.transforms as transforms       # 导入图像预处理工具
import cv2                                        # 导入 OpenCV，用于读取和解码视频帧
import numpy as np                                # 导入 NumPy，用于将提取的特征保存为 .npy 文件
import os                                         # 用于处理文件路径和文件夹操作
import json                                       # 用于读取视频列表配置
import yaml                                       # 用于读取项目的总配置文件

# ==========================================
# 主函数：提取全局特征
# ==========================================
def extract_global_features(config_path="configs/config_teacher.yaml", video_root="data/Teacher/raw_videos/"):
    # 1. 读取 YAML 配置
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    raw_json_path = config['data']['raw_annotation'] # 获取所有视频 ID 的列表文件路径
    output_dir = config['data']['feature_dir']       # 获取我们要保存 .npy 特征的目标文件夹
    os.makedirs(output_dir, exist_ok=True)           # 创建目标文件夹
    
    # 自动检测硬件：如果有苹果芯片 (MPS) 就用 MPS 加速，否则用 CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # ---------------------------------------------------------
    # 第一部分：加载 2D 网络 (ResNet-101) - 用于提取单帧画面的“静态背景外观特征” (如：天空、草地、房间)
    # ---------------------------------------------------------
    print(f"Loading ResNet-101 (2D) on {device}...")
    resnet = models.resnet101(pretrained=True)       # 下载并加载在 ImageNet 上训练好的 ResNet-101
    # 关键操作：我们不需要 ResNet 最后的 1000 类分类器，所以用 children()[:-1] 把它砍掉，只保留特征提取层
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])).to(device)
    resnet.eval()                                    # 设为评估模式，关闭 Dropout 和 BatchNorm 的更新
    
    # ---------------------------------------------------------
    # 第二部分：加载 3D 网络 (R3D_18) - 用于提取多帧连起来的“动态动作特征” (如：跑、跳、切菜)
    # ---------------------------------------------------------
    print(f"Loading R3D_18 (3D Motion CNN) on {device}...")
    c3d_model = models.video.r3d_18(pretrained=True) # R3D_18 是原论文 C3D 模型的现代完美平替
    # 同样砍掉最后的分类层，获取 512 维的纯粹动作特征
    c3d_model = torch.nn.Sequential(*(list(c3d_model.children())[:-1])).to(device)
    c3d_model.eval()

    # ---------------------------------------------------------
    # 第三部分：定义图像预处理流水线 (Preprocessing Pipelines)
    # ---------------------------------------------------------
    # ResNet-101 要求的标准预处理：224x224 分辨率
    preprocess_2d = transforms.Compose([
        transforms.ToPILImage(),                     # 将 OpenCV 读取的 NumPy 数组转为 PIL 图像
        transforms.Resize(256),                      # 先缩放到 256x256
        transforms.CenterCrop(224),                  # 然后在正中间裁剪出 224x224 (防止画面拉伸变形)
        transforms.ToTensor(),                       # 转换为 [0, 1] 范围的 PyTorch Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet 官方统计的均值和标准差归一化
    ])
    
    # R3D_18 要求的标准预处理：112x112 分辨率 (3D 网络很吃显存，所以分辨率要求更小)
    preprocess_3d = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.432, 0.394, 0.376], std=[0.228, 0.224, 0.225]), # Kinetics-400 数据集的标准归一化
    ])
    
    num_frames = config['model']['num_frames']       # 从每个视频中均匀抽取的帧数 (通常是 16 帧)
    
    # ---------------------------------------------------------
    # 第四部分：遍历视频文件，逐个提取特征
    # ---------------------------------------------------------
    with open(raw_json_path, 'r') as f:
        msvd_data = json.load(f)
        
    for item in msvd_data:
        vid_id = item['video_id']
        video_path = os.path.join(video_root, item.get('video_path', f"{vid_id}.avi")) # 拼接视频绝对路径
            
        if not os.path.exists(video_path):           # 如果硬盘上没这个视频，跳过
            continue
            
        cap = cv2.VideoCapture(video_path)           # 使用 OpenCV 打开视频文件
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 获取视频总帧数
        if total_frames == 0: 
            cap.release()
            continue
            
        # np.linspace：在 [0, total_frames-1] 范围内均匀生成 16 个数字，代表我们要抽取的帧的索引
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        tensors_2d = []                              # 用于存放 2D 处理后的帧
        tensors_3d = []                              # 用于存放 3D 处理后的帧
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)    # 让 OpenCV 跳转到指定帧的位置
            ret, frame = cap.read()                  # 读取那一帧的画面
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # OpenCV 默认是 BGR，深度学习模型要 RGB，需要翻转颜色通道
                tensors_2d.append(preprocess_2d(frame_rgb))        # 执行 2D 预处理并存入列表
                tensors_3d.append(preprocess_3d(frame_rgb))        # 执行 3D 预处理并存入列表
        cap.release()                                # 释放视频文件句柄
        
        # 异常处理：如果视频太短导致抽出来的帧不足 16 帧，用全黑的纯零张量补齐 (Padding)
        while len(tensors_2d) < num_frames:
            tensors_2d.append(torch.zeros((3, 224, 224)))
            tensors_3d.append(torch.zeros((3, 112, 112)))
            
        # ==========================================
        # 第五部分：特征张量的维度拼接与网络推理
        # ==========================================
        # 将 16 张独立的图片堆叠成一个 Batch：形状变为 [16, 3, 224, 224] = [Batch_size, Channels, Height, Width]
        batch_2d = torch.stack(tensors_2d).to(device)
        
        # 3D 卷积的输入形状要求非常特殊：[Batch, Channels, Time, Height, Width]
        # torch.stack 后是 [16, 3, 112, 112]，需要 permute 把 16 移到中间变成 [3, 16, 112, 112]
        # 然后 unsqueeze(0) 在最前面增加一维 Batch=1，最终变为 [1, 3, 16, 112, 112]
        batch_3d = torch.stack(tensors_3d).permute(1, 0, 2, 3).unsqueeze(0).to(device)
        
        # with torch.no_grad()：特征提取不需要计算梯度，极大节省显存和加速
        with torch.no_grad():
            # 1. 2D 特征处理
            feat_2d = resnet(batch_2d).squeeze()     # 将 16 帧送入 ResNet，输出形状 [16, 2048]
            feat_2d_mean = feat_2d.mean(dim=0)       # 沿着时间维度(16帧)求平均，融合为整段视频的 1 个特征，形状 [2048]
            # 因为原论文规定的特征是 512 维，这里用一维自适应平均池化(adaptive_avg_pool1d)强行把 2048 维压缩成 512 维
            feat_2d_512 = torch.nn.functional.adaptive_avg_pool1d(feat_2d_mean.unsqueeze(0).unsqueeze(0), 512).squeeze()
            
            # 2. 3D 特征处理
            # 3D CNN 本身就会处理时间维度，所以直接输出的就是整段视频的动作特征，刚好是完美的 512 维
            feat_3d_512 = c3d_model(batch_3d).squeeze() 
            
            # 3. 特征融合
            # 将 2D(外观512维) 和 3D(动作512维) 拼接在一起，变成一个 1024 维的超级向量
            global_feat = torch.cat([feat_2d_512, feat_3d_512], dim=0).cpu().numpy()
            
        # ==========================================
        # 第六部分：保存结果
        # ==========================================
        # 以 NumPy 格式保存到硬盘上，比如 "vid1_global.npy"
        np.save(os.path.join(output_dir, f"{vid_id}_global.npy"), global_feat)
        print(f"Processed 2D+3D(C3D-equivalent) features for {vid_id}")

if __name__ == "__main__":
   extract_global_features(config_path="configs/config_teacher.yaml", video_root="data/Teacher/raw_videos/")