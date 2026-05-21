import torch                                      # 导入 PyTorch 核心库
import torchvision                                # 导入 torchvision，主要为了使用里面的 roi_align 算子
import cv2                                        # OpenCV，用于读取视频帧
import numpy as np                                # 处理数组和保存 .npy 文件
import os                                         # 处理文件和路径
import json                                       # 读取 JSON 标注文件
import yaml                                       # 读取 YAML 配置文件

# ==========================================
# 主函数：使用 YOLOv5 提取每一帧的局部物体特征和坐标
# ==========================================
def extract_yolo_roi(config_path="configs/config_teacher.yaml", video_root="data/Teacher/raw_videos/"):
    # 1. 读取配置文件
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_json_path = config['data']['raw_annotation']  # 
    output_dir = config['data']['feature_dir']        # 特征保存目录
    os.makedirs(output_dir, exist_ok=True)            # 确保输出文件夹存在

    # 检测硬件设备（Mac 芯片使用 MPS）
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading YOLOv5 on {device}...")
    
    # 2. 动态加载 YOLOv5 模型
    # 从 PyTorch Hub 在线/缓存加载轻量级的 yolov5s，pretrained=True 表示使用 COCO 数据集预训练权重
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
    yolo.eval()                                      # 设置为评估模式
    
    num_frames = config['model']['num_frames']       # 需要提取的帧数，默认 16 帧
    top_n = config['model']['num_objects']           # 每帧最多保留的物体数量，STGIN 默认是 5 个物体
    
    with open(raw_json_path, 'r') as f:
        msvd_data = json.load(f)
        
    # ---------------------------------------------------------
    # 开始遍历每个视频
    # ---------------------------------------------------------
    for item in msvd_data:
        vid_id = item['video_id']
        # 拼接视频路径，如果 JSON 里有 video_path 字段就用，否则默认叫 vid_id.avi
        video_path = os.path.join(video_root, item.get('video_path', f"{vid_id}.avi"))
        
        # （被注释掉的代码：如果已经提取过，就跳过。用于断点续传）
        # if os.path.exists(os.path.join(output_dir, f"{vid_id}_roi.npy")):
        #     continue
            
        if not os.path.exists(video_path):           # 硬盘上找不到视频则跳过
            print(f"Warning: Not found {video_path}, skipping {vid_id}")
            continue
            
        cap = cv2.VideoCapture(video_path)           # OpenCV 打开视频
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: 
            cap.release()
            continue
            
        # 在视频总长度中均匀采样 16 个时间点
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)    # 指针跳到指定帧
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # 转 RGB
        cap.release()
        
        # 视频太短的 padding 机制，用黑图补齐到 16 帧
        while len(frames) < num_frames:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
        roi_features_list = []                       # 用于存放这一整个视频的所有 ROI 特征
        coords_list = []                             # 用于存放这一整个视频的所有 坐标位置
        
        # ---------------------------------------------------------
        # 对采样的每一帧，使用 YOLO 寻找物体并提取区域特征
        # ---------------------------------------------------------
        for frame in frames:
            results = yolo(frame)                    # 把图片喂给 YOLOv5 进行目标检测
            
            # 使用 pandas 格式获取检测结果表格，包含 [xmin, ymin, xmax, ymax, confidence, class, name]
            df = results.pandas().xyxy[0]            
            # 过滤掉置信度 < 0.5 的杂框（保证检测到的都是清晰明确的物体）
            df = df[df['confidence'] >= 0.5]         
            # 按置信度从高到低排序，并强制只截取前 5 个 (top_n) 框
            df = df.sort_values(by='confidence', ascending=False).head(top_n)
            
            # 将 numpy 图片 [H, W, C] 转换为 PyTorch 需要的 [Batch, C, H, W]，并归一化到 0~1
            # 这里 Batch = 1，所以是 unsqueeze(0)
            frame_tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            # 构建 torchvision.ops.roi_align 需要的 boxes 格式
            boxes = []
            for _, row in df.iterrows():
                # ROI Align 的框格式为 [batch_index, x1, y1, x2, y2]
                # 因为只有 1 张图片，所以 batch_index 永远是 0
                boxes.append([0, row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                
            # 万一画面很干净，YOLO 连 5 个物体都没找到，用 [0,0,0,0,0] 纯零占位符补齐，保证矩阵维度严格对齐
            while len(boxes) < top_n:
                boxes.append([0, 0, 0, 0, 0])
                
            boxes_tensor = torch.tensor(boxes).float() # 转为张量
            
            # === 核心创新：直接在原图上做 ROI Align ===
            # 将含有物体的区域画面抠出来，强制缩放(Align)为 7x7 大小的小图
            # output_size=(7, 7)，出来的形状是 [5, 3通道, 7, 7]
            roi_out = torchvision.ops.roi_align(frame_tensor.cpu(), boxes_tensor.cpu(), output_size=(7, 7))
            
            # 展平特征：把 3*7*7=147 个像素值展平为一维向量，形状变为 [5, 147]
            roi_feat = roi_out.view(top_n, -1) 
            
            # === 维度对齐 STGIN ===
            # STGIN 的全连接层(GCN)要求每个物体是 512 维特征
            if roi_feat.shape[-1] > 512:             # 如果大于 512 (在此代码中不会发生)，就截断
                roi_feat = roi_feat[:, :512]
            else:
                # 因为我们是从原图扣的 RGB，只有 147 维。所以我们在后面疯狂补 0，强行撑满 512 维！
                # 这是在 Mac 算力受限下，为了跑通庞大网络而做出的工程妥协（Padding 补齐法）
                roi_feat = torch.cat([roi_feat, torch.zeros(top_n, 512 - roi_feat.shape[-1])], dim=-1)
                
            # 保存这 1 帧的 5 个物体的特征和坐标（去掉 batch_index 那一列）
            roi_features_list.append(roi_feat.numpy())
            coords_list.append(boxes_tensor[:, 1:].numpy())
            
        # ---------------------------------------------------------
        # 保存整个视频的数据
        # ---------------------------------------------------------
        # roi_features_list 拼成数组，形状：[16帧, 5个物体, 512维]
        np.save(os.path.join(output_dir, f"{vid_id}_roi.npy"), np.array(roi_features_list))
        # coords_list 拼成数组，形状：[16帧, 5个物体, 4维坐标(x1,y1,x2,y2)]
        np.save(os.path.join(output_dir, f"{vid_id}_coords.npy"), np.array(coords_list))
        
        print(f"Processed YOLO features for {vid_id}")

if __name__ == "__main__":
    # 执行脚本
    extract_yolo_roi(config_path="configs/config_teacher.yaml", video_root="data/Teacher/raw_videos/")