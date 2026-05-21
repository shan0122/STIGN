import torch                                      # 导入 PyTorch 核心库
import torch.optim as optim                       # 导入优化器模块 (用于更新模型权重)
import yaml                                       # 用于读取 yaml 配置文件
import json                                       # 用于读取 json 词表文件
from dataset.dataloader import get_dataloader     # 导入我们之前写好的数据加载器
from models.stgin import STGIN                    # 导入模型架构
from utils.loss import stgin_loss                 # 导入自定义的联合损失函数

# ==========================================
# 辅助函数：生成因果掩码 (Causal Mask)
# ==========================================
def generate_causal_mask(sz):
    # Transformer Decoder 训练时的核心机制：
    # 比如输入是 "a man is"，输出要预测 "playing"。模型在看到 "is" 时，绝不能偷看到后面的 "playing"
    # 生成一个下三角矩阵，未来不可见的词用 -inf 遮挡，可见的词用 0 保留
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

# ==========================================
# 训练主循环
# ==========================================
def main():
    # 1. 读取配置文件
    with open("configs/config_teacher.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # 2. 自动检测设备 (Mac 苹果芯片优先使用 mps 加速，否则用 cpu)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 3. 读取词表文件，获取词典的大小 (用于构建模型的 Embedding 层和最后的分类器维度)
    with open(config['data']['vocab_path'], 'r') as f:
        vocab_size = len(json.load(f))
        
    # 4. 获取训练集的数据迭代器
    train_loader = get_dataloader(config, split='train')
    
    # 5. 实例化 STGIN 模型，并将它移动到指定的运算设备 (如 GPU/MPS) 上
    model = STGIN(
        vocab_size=vocab_size, 
        feature_dim=config['model']['feature_dim'],
        num_semantic_concepts=config['model']['num_semantic_concepts'],
        num_layers=config['model']['transformer_layers'],
        num_heads=config['model']['transformer_heads'],
        mu=config['model']['mu']                           # GCN 空间邻接图的距离阈值参数
    ).to(device)
    
    # 6. 定义优化器
    # 使用最经典、在 NLP/CV 任务中最常用的 Adam 优化器
    # 我们故意移除了 weight_decay (L2 正则化)，因为复杂的 Transformer 模型加了 L2 容易导致欠拟合
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'])
    
    # 读取配置：长度惩罚系数 (beta) 和 梯度累积步数 (accum_steps)
    beta = config['train']['beta']
    accum_steps = config['train']['grad_accum_steps']
    
    # 初始化最优损失为一个极大值
    best_loss = float('inf')
    
    # ==========================================
    # 开始 Epoch 循环 (完整遍历数据集的次数)
    # ==========================================
    for epoch in range(config['train']['epochs']):
        model.train()                                      # 将模型设置为训练模式 (开启 Dropout 等)
        total_loss = 0                                     # 用于累计当前 epoch 的总损失
        optimizer.zero_grad()                              # 在每一个 epoch 开始前，清空旧的梯度
        
        # 遍历数据加载器，每次吐出一个 Batch (如 32 个视频样本)
        for batch_idx, batch in enumerate(train_loader):
            # 将所有数据从内存(CPU)转移到显存(GPU/MPS)上
            roi_feats = batch['roi_feats'].to(device)
            coords = batch['coords'].to(device)
            global_feats = batch['global_feats'].to(device)
            tokens = batch['tokens'].to(device)
            semantic_labels = batch['semantic_labels'].to(device)
            
            # --- 关键的 Teacher Forcing 序列错位操作 ---
            # tgt_input 是模型的输入：取除了最后一个词之外的所有词 (如: <BOS> a man is)
            # tgt_target 是模型的预测目标：取除了第一个词之外的所有词 (如: a man is playing)
            tgt_input, tgt_target = tokens[:, :-1], tokens[:, 1:]
            
            # 生成对应长度的因果掩码
            tgt_mask = generate_causal_mask(tgt_input.size(1)).to(device)
            
            # 7. 前向传播 (Forward Pass)
            # 拿到预测的词汇概率 (logits) 和 语义标签概率 (semantic_probs)
            logits, semantic_probs = model(roi_feats, coords, global_feats, tgt_input, tgt_mask)
            
            # 8. 计算损失
            loss = stgin_loss(logits, tgt_target, semantic_probs, semantic_labels, beta=beta)
            
            # --- 显存拯救神技：梯度累积 (Gradient Accumulation) ---
            # 如果你的显存不够大，batch_size 只能设得很小（如 8），梯度方向会非常抖动。
            # accum_steps=4 的意思是：我们将 loss 除以 4，连续反向传播 4 次才更新一次权重。
            # 这样相当于用极小的显存，模拟出了 8 * 4 = 32 的大 Batch Size 效果！
            loss = loss / accum_steps
            
            # 9. 反向传播 (Backward Pass)，计算每个权重的梯度
            loss.backward()
            
            # 当累积的批次达到了设定的步数，或者是这个 epoch 的最后一个 batch 时，执行权重更新
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                # 10. 梯度裁剪 (Gradient Clipping)
                # Transformer 架构在训练早期非常容易出现“梯度爆炸”导致 loss 变成 NaN
                # 这句话限制了梯度的最大范数不超过 1.0，保证训练过程极其稳定
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()                           # 根据梯度，走一步更新神经网络的所有参数
                optimizer.zero_grad()                      # 更新完后，立刻清空梯度，迎接下一轮累积
                
            # 因为之前 loss 除以了 accum_steps，为了打印出真实的 loss 大小，这里再乘回来
            total_loss += loss.item() * accum_steps
            
            # 11. 打印训练进度 (每处理 50 个 batch 打印一次)
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{config['train']['epochs']}], Step [{batch_idx}], Loss: {loss.item() * accum_steps:.4f}")
                
        # 一个 epoch 结束后，计算该 epoch 的平均损失
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}] Avg Loss: {avg_loss:.4f}")
        
        # 12. 保存表现最好的模型 (Early Stopping 策略的体现)
        # 如果当前的平均损失打破了历史最低记录，则将当前的权重保存下来
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 将模型权重字典序列化，保存为 .pth 文件。这个文件就是你训练出的心血结晶！
            torch.save(model.state_dict(), "stgin_teacher_best.pth")
            print(f"--> Saved new best model at epoch {epoch+1}")

if __name__ == "__main__":
    main()                                                 # 执行训练主函数