import torch
import torch.nn.functional as F

# ==========================================
# STGIN 联合损失函数
# 包含：语义检测损失 (SCD Loss) + 描述生成损失 (Caption Loss)
# 参数 beta: 句子长度惩罚的超参数，默认为 0.4
# ==========================================
def stgin_loss(logits, targets, semantic_probs, semantic_targets, beta=0.4):
    
    # ---------------------------------------------------------
    # 1. 语义概念检测损失 (Semantic Concept Detection Loss)
    # ---------------------------------------------------------
    # 使用二元交叉熵 (Binary Cross Entropy)。
    # 因为每个视频可能包含多个标签（比如既有 'man' 又有 'guitar'），
    # 这属于多标签分类任务，不能用 Softmax+CE，必须用 Sigmoid+BCE。
    loss_scd = F.binary_cross_entropy(semantic_probs, semantic_targets)
    
    # ---------------------------------------------------------
    # 2. 描述生成损失 (Caption Generation Loss)
    # ---------------------------------------------------------
    # 解析张量维度：B 是 Batch 大小，SeqLen 是句子长度(26)，VocabSize 是词表大小
    B, SeqLen, VocabSize = logits.shape
    
    # 为了使用 PyTorch 内置的 CrossEntropy，必须将张量展平
    # 把所有 batch 和所有时间步的预测结果揉在一起：变成 [(B * SeqLen), VocabSize]
    logits = logits.reshape(-1, VocabSize)
    # 把真实的目标句子也展平：变成 [B * SeqLen]
    targets = targets.reshape(-1)
    
    # 计算标准的交叉熵损失 (Cross Entropy)。
    # ignore_index=0：这是自然语言处理最重要的设置！遇到真实标签是 0 (<PAD>) 的地方，
    # 强制不计算损失、不传播梯度。因为预测填充词是没有意义的。
    # reduction='none'：不要自动求平均或求和，保留每个单词的独立损失值。
    loss_ce = F.cross_entropy(logits, targets, ignore_index=0, reduction='none')
    
    # 将长条状的损失数组重新变回二维矩阵 [Batch, SeqLen]
    # 这样矩阵的每一行就是“一句话里每个单词的损失”
    loss_ce = loss_ce.reshape(B, SeqLen)
    
    # ---------------------------------------------------------
    # 3. 计算长度惩罚 (Length Penalty)
    # ---------------------------------------------------------
    # 统计每句话里的真实单词数量（即除去 0 以外的单词个数）
    # .sum(dim=1) 对每一行求和，得到形状 [Batch] 的向量
    seq_lengths = (targets.reshape(B, SeqLen) != 0).sum(dim=1).float()
    
    # torch.clamp 将最小长度强制设为 1，防止后面数学运算时出现除以 0 的错误
    seq_lengths = torch.clamp(seq_lengths, min=1.0)
    
    # 计算长度惩罚系数：长度的 -beta 次方 (即 1 / 长度^beta)
    # 为什么需要这个？因为长句子的单词多，损失累加起来自然比短句子大。
    # 如果不除以长度，模型会“偷懒”，疯狂输出极短的句子来骗取较低的损失。
    length_penalty = torch.pow(seq_lengths, -beta)
    
    # ---------------------------------------------------------
    # 4. 最终损失合成
    # ---------------------------------------------------------
    # loss_ce.sum(dim=1)：把一句话里所有单词的损失加起来
    # * length_penalty：乘上惩罚系数，压制“短句作弊”现象
    # .mean()：最后对整个 Batch 的句子求平均损失
    loss_caption = (loss_ce.sum(dim=1) * length_penalty).mean()
    
    # 将多标签检测损失和字幕生成损失直接相加（权重 1:1），返回总损失
    return loss_scd + loss_caption