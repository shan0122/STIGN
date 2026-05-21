import torch                 # 导入 PyTorch 基础库
import torch.nn as nn        # 导入神经网络模块 (包含各种层如 Linear, Conv 等)
import torch.nn.functional as F # 导入无参数的函数接口 (如 relu, softmax, normalize)

# ==========================================
# 1. 空间时空图交互模块 (TSI_Module)
# 作用：在单帧画面内，建立多个目标物体之间的空间和外观关系图。
# ==========================================
class TSI_Module(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=512, mu=0.5):
        super().__init__()
        # 定义一个全连接层，用于图卷积网络(GCN)中的特征变换
        self.W_gcn = nn.Linear(feature_dim, hidden_dim)
        # 空间距离的阈值，用于判定两个物体是否在空间上产生连接
        self.mu = mu
        
    def forward(self, roi_features, coords):
        # B:Batch尺寸, T:帧数, N:每帧目标数(比如5), D:特征维度(512)
        B, T, N, D = roi_features.shape
        
        # 沿着特征维度进行 L2 归一化，防止特征绝对值影响相似度计算
        norm_features = F.normalize(roi_features, p=2, dim=-1)
        # 计算物体间的外观相似度矩阵 (N个物体两两内积)
        appearance_adj = torch.matmul(norm_features, norm_features.transpose(-1, -2))
        # 找到每个物体与其他物体最大相似度，加上1e-8防除零错误
        max_adj = torch.max(appearance_adj, dim=-1, keepdim=True)[0] + 1e-8
        # 将外观邻接矩阵归一化到 [0, 1] 之间，得到外观关系图 R_a
        R_a = appearance_adj / max_adj  
        
        # 根据框的坐标 [x_min, y_min, x_max, y_max] 计算中心点 X 坐标
        center_x = (coords[..., 0] + coords[..., 2]) / 2
        # 计算中心点 Y 坐标
        center_y = (coords[..., 1] + coords[..., 3]) / 2
        # 将 X 和 Y 坐标堆叠成 (B, T, N, 2) 的中心点矩阵
        centers = torch.stack([center_x, center_y], dim=-1)
        # 计算N个中心点之间的欧氏距离(L2距离)矩阵
        dist_matrix = torch.cdist(centers, centers, p=2)
        # 如果距离小于阈值 mu，则为1(相连)，否则为0。得到空间关系图 R_s
        R_s = (dist_matrix < self.mu).float()
        
        # 融合外观和空间关系，利用 ReLU 去除负值，得到最终邻接矩阵 G
        G = F.relu(R_a * R_s)
        # 计算图中每个节点的度(Degree)，即与它相连的权重之和
        degree = torch.sum(G, dim=-1, keepdim=True) + 1e-8
        # 对度矩阵求负半偏方（图卷积网络 GCN 的标准对称归一化步骤）
        degree_inv_sqrt = torch.pow(degree, -0.5)
        # 得到对称归一化后的图邻接矩阵 G_norm = D^{-1/2} * G * D^{-1/2}
        G_norm = degree_inv_sqrt * G * degree_inv_sqrt.transpose(-1, -2)
        
        # 对原始节点特征进行线性变换
        node_features = self.W_gcn(roi_features)
        # 节点特征在图上进行消息传递(聚集邻居特征)
        updated_features = torch.matmul(G_norm, node_features)
        # 残差连接(Residual)并经过激活函数，得到图卷积后的特征
        out_features = F.relu(roi_features + updated_features)
        # 将一帧内的N个物体特征求平均，得到每一帧的全局空间交互特征
        return torch.mean(out_features, dim=2)

# ==========================================
# 2. 轨迹时间上下文模块 (TTC_Module)
# 作用：捕捉同一个或相关物体在不同时间帧（时序）上的运动轨迹。
# ==========================================
class TTC_Module(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        # 用于生成注意力机制中的 Key 和 Value 特征
        self.W_t = nn.Linear(feature_dim, feature_dim)
        # 用于生成注意力机制中的 Query 特征
        self.W_r = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, object_features):
        B, T, N, D = object_features.shape
        # 提取每一帧的第1个物体(通常是面积最大或置信度最高的)作为向导(Guide)特征
        guide_features = object_features[:, :, 0, :] 
        # 对向导特征进行线性变换得到 Query (q)
        q = self.W_r(guide_features) 
        # 将所有帧的所有物体特征展平为 (B, T*N, D)，并线性变换得到 Key (k_flat)
        k_flat = self.W_t(object_features).view(B, T * N, D)
        
        # 计算 q 和 k_flat 的点积注意力分数，除以根号 D 进行缩放(Scaled Dot-Product)
        scores = torch.matmul(q, k_flat.transpose(1, 2)) / (D ** 0.5)
        # 使用 Softmax 将分数转化为概率分布（注意力权重）
        attention_weights = F.softmax(scores, dim=-1) 
        # 使用注意力权重对所有物体特征进行加权求和，得到具有时序上下文的特征
        temporal_features = torch.matmul(attention_weights, object_features.view(B, T * N, D))
        return temporal_features

# ==========================================
# 3. 语义概念检测模块 (SCD_Module)
# 作用：从全局视觉特征中预测视频包含的高频属性词（如 dog, run, car 等）。
# ==========================================
class SCD_Module(nn.Module):
    def __init__(self, global_feature_dim=1024, num_semantic_concepts=300):
        super().__init__()
        # 定义一个多层感知机 (MLP)
        self.mlp = nn.Sequential(
            # 全连接层将特征映射到 1024 维
            nn.Linear(global_feature_dim, 1024),
            # ReLU 激活函数增加非线性
            nn.ReLU(),
            # Dropout 随机丢弃50%的神经元，防止过拟合
            nn.Dropout(0.5),
            # 映射到指定数量的语义概念类别 (如300类)
            nn.Linear(1024, num_semantic_concepts), 
            # Sigmoid 将输出压缩到 0-1 之间，代表每个概念存在的概率
            nn.Sigmoid()
        )
    def forward(self, global_visual_features):
        # 直接前向传播返回每个语义单词的概率分布
        return self.mlp(global_visual_features)

# ==========================================
# 4. 描述生成器模块 (DescriptionGenerator)
# 作用：基于融合后的记忆特征(Memory)和已生成的单词，预测下一个单词。
# ==========================================
class DescriptionGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads):
        super().__init__()
        # 词嵌入层，将单词索引转化为词向量
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        # 定义 Transformer 解码器的一层结构
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=2048, batch_first=True)
        # 堆叠多层 Transformer 解码器层
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # 最终的输出分类层，映射回词表大小
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, tgt_seq, memory_features, tgt_mask=None):
        # 将输入的单词序列转为词向量
        tgt_embed = self.word_embed(tgt_seq)
        # Transformer 解码器：接收目标词向量(tgt)和视觉特征(memory)，以及因果掩码(tgt_mask)
        out = self.transformer_decoder(tgt=tgt_embed, memory=memory_features, tgt_mask=tgt_mask)
        # 通过全连接层输出预测下一个单词的逻辑值 (Logits)
        return self.fc_out(out)

# ==========================================
# 5. 主网络组装模块 (STGIN)
# 作用：将上述四个子模块组合拼装，形成最终的端到端大模型。
# ==========================================
class STGIN(nn.Module):
    def __init__(self, vocab_size, feature_dim, num_semantic_concepts, num_layers, num_heads, mu):
        super().__init__()
        # 实例化空间交互模块
        self.tsi = TSI_Module(feature_dim, feature_dim, mu)
        # 实例化时序上下文模块
        self.ttc = TTC_Module(feature_dim)
        # 实例化语义概念预测模块 (输入维度是 feature_dim*2，因为 R3D 特征通常很大)
        self.scd = SCD_Module(feature_dim * 2, num_semantic_concepts)
        # 实例化最终的语言生成器模块
        self.generator = DescriptionGenerator(vocab_size, feature_dim, num_layers, num_heads)
        # 将 SCD 输出的 300 维概率向量，再映射回标准的 512 维特征，以便于拼接
        self.semantic_proj = nn.Linear(num_semantic_concepts, feature_dim)
        
    def forward(self, roi_features, coords, global_features, tgt_seq, tgt_mask=None):
        # 1. 经过 TSI 模块，得到空间交互特征 (B, T, D)
        spatial_feats = self.tsi(roi_features, coords)
        # 2. 经过 TTC 模块，得到时间上下文特征 (B, T, D)
        temporal_feats = self.ttc(roi_features)
        # 3. 经过 SCD 模块，得到语义单词存在的概率 (B, num_concepts)
        semantic_probs = self.scd(global_features)
        # 4. 将语义概率转化为 512 维的视觉-语义特征，并增加时间维度 (B, 1, D)
        semantic_feats = self.semantic_proj(semantic_probs).unsqueeze(1)
        
        # 5. 将空间特征、时间特征、语义特征在时间序列维度(dim=1)上拼接到一起
        memory_features = torch.cat([spatial_feats, temporal_feats, semantic_feats], dim=1)
        # 6. 将拼接好的综合记忆特征和当前的单词输入给生成器，输出下一个词的预测
        logits = self.generator(tgt_seq, memory_features, tgt_mask)
        
        # 返回预测单词的分布 logits，以及辅助训练用的语义属性概率 semantic_probs
        return logits, semantic_probs