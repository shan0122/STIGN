import torch                                      # 导入 PyTorch 核心库
import yaml                                       # 导入 YAML 库，用于读取配置文件
import json                                       # 导入 JSON 库，用于读取词表和测试集标签
import os                                         # 导入 OS 库，用于处理文件路径
import random                                     # 导入 random 库，用于随机抽样展示结果
from tqdm import tqdm                             # 导入 tqdm，用于在终端显示进度条
from models.stgin import STGIN                    # 从你的模型文件中导入 STGIN 网络架构
from dataset.dataloader import get_dataloader     # 导入数据加载器函数

# ==========================================
# 尝试导入标准的自然语言生成评测工具包 (COCO Evaluation)
# ==========================================
try:
    from pycocoevalcap.bleu.bleu import Bleu      # 导入 BLEU 评测指标（评估 N-gram 准确率）
    from pycocoevalcap.rouge.rouge import Rouge   # 导入 ROUGE 评测指标（评估召回率和流畅度）
    from pycocoevalcap.cider.cider import Cider   # 导入 CIDEr 评测指标（专门针对图像/视频描述的指标）
except ImportError:
    # 如果没安装这个库，打印提示信息
    print("Please install pycocoevalcap: pip install pycocoevalcap")

# ==========================================
# 函数：生成因果掩码 (Causal Mask)
# 作用：在 Transformer 解码时，遮蔽掉未来的词，防止模型“偷看”答案
# ==========================================
def generate_causal_mask(sz):
    # 生成一个上三角全为 1 的矩阵，然后转置，得到下三角矩阵
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    # 将 0 的地方替换为负无穷(-inf)，1 的地方替换为 0.0，传给注意力层后，-inf 处权重会变成 0
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

# ==========================================
# 函数：自回归解码 (这里实际上是贪心搜索 Greedy Search)
# 作用：模型一步一步，一个词一个词地把整句话生成出来
# ==========================================
def beam_search(model, roi_feats, coords, global_feats, config, device):
    model.eval()                                  # 切换模型为评估模式，关闭 Dropout
    max_len = 30                                  # 设定生成的句子最长不能超过 30 个词
    bos = 1                                       # 词表中 <BOS> (句首) 的索引是 1
    eos = 2                                       # 词表中 <EOS> (句尾) 的索引是 2
    
    seq = [bos]                                   # 初始化生成的序列，先把 <BOS> 放进去
    with torch.no_grad():                         # 推理阶段不需要计算梯度，节省显存加速计算
        for _ in range(max_len):                  # 开始循环生成，最多生成 max_len 次
            # 将当前已经生成的序列转化为 Tensor，形状 [1, 当前长度]
            tgt_tensor = torch.tensor([seq]).to(device)
            # 根据当前序列长度生成掩码矩阵
            tgt_mask = generate_causal_mask(tgt_tensor.size(1)).to(device)
            
            # 将视觉特征和当前序列喂给模型，模型会输出序列下一个词的概率分布 logits
            logits, _ = model(roi_feats, coords, global_feats, tgt_tensor, tgt_mask)
            
            # 取出序列最后一个位置的输出 [1(Batch), -1(最后一个时间步), VocabSize]
            next_word_logits = logits[0, -1, :] 
            # 贪心策略：直接挑选概率最高的那一个词的索引作为生成的下一个词
            next_word = next_word_logits.argmax(dim=-1).item()
            
            # 如果模型输出的是句尾符号 <EOS>，说明它觉得话说完了，跳出循环停止生成
            if next_word == eos:
                break
            # 将新生成的词追加到序列末尾，参与下一轮循环的输入
            seq.append(next_word)
            
    return seq                                    # 返回最终生成的单词索引列表

# ==========================================
# 主测试函数
# ==========================================
def test():
    # 1. 读取配置文件
    with open("configs/config_teacher.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # 设置设备 (优先使用 Mac 的 MPS 加速)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 2. 读取词表，并构建从“数字索引”反向映射回“英文单词”的字典 idx2word
    with open(config['data']['vocab_path'], 'r') as f:
        vocab = json.load(f)
    idx2word = {idx: word for word, idx in vocab.items()}
    vocab_size = len(vocab)
    
    # 3. 初始化 STGIN 模型结构
    model = STGIN(
        vocab_size=vocab_size, 
        feature_dim=config['model']['feature_dim'],
        num_semantic_concepts=config['model']['num_semantic_concepts'],
        num_layers=config['model']['transformer_layers'],
        num_heads=config['model']['transformer_heads'],
        mu=config['model']['mu']
    ).to(device)

    # 4. 加载你在训练阶段保存的最好权重 (stgin_teacher_best.pth)
    if os.path.exists("stgin_teacher_best.pth"):
        model.load_state_dict(torch.load("stgin_teacher_best.pth", map_location=device))
        print("Loaded best model weights.")
    else:
        print("Warning: Weight file stgin_teacher_best.pth not found. Testing with random weights.")
    model.eval()                                  # 切换为评估模式
    
    gts = {}  # ground truths: 用于存放每个视频对应的人类真实答案
    res = {}  # results: 用于存放每个视频由模型生成的答案
    
    # 5. 读取测试集的原始标注文件
    with open(os.path.join(config['data']['processed_dir'], 'teacher_test.json'), 'r') as f:
        test_annotations = json.load(f)
        
    # 提取真实答案：将每个视频的真实单词索引列表，翻译回英文字符串，存入 gts 字典
    for item in test_annotations:
        vid = item['video_id']
        # 列表推导式：过滤掉 0,1,2 (<PAD>,<BOS>,<EOS>)，把中间的数字翻译成单词并用空格拼成句子
        caps = [" ".join([idx2word.get(idx, '<UNK>') for idx in seq if idx not in (0, 1, 2)]) for seq in item['tokens_list']]
        gts[vid] = caps
    
    # 6. 获取测试集的数据加载器 DataLoader
    test_loader = get_dataloader(config, split='test')
    
    # 7. 开始在测试集上进行模型推理
    print("Running Inference...")
    for batch in tqdm(test_loader):               # 遍历所有的测试集 batch
        # 提取 Batch 中的第一个样本特征并转移到显卡 (这里 batch_size 虽然可能是 32，但推断为了简便和安全，每次只取第 1 个[0:1])
        roi = batch['roi_feats'][0:1].to(device)
        coords = batch['coords'][0:1].to(device)
        glb = batch['global_feats'][0:1].to(device)
        vid_id = batch['video_id'][0] 
        
        # 如果这个视频已经生成过了，就跳过（因为测试集中同一个视频可能被存了多条样本）
        if vid_id in res: continue 
        
        # 调用前面的 beam_search 函数，让模型生成一句话
        best_seq = beam_search(model, roi, coords, glb, config, device)
        
        # 将生成的数字索引翻译回英文句子
        gen_words = [idx2word.get(idx, '<UNK>') for idx in best_seq if idx not in (0, 1, 2)]
        res[vid_id] = [" ".join(gen_words)]       # 存入结果字典 res
        
    # # =====================================================
    # # 定性分析：随机打印 5 个模型生成的句子和真实答案对比
    # # =====================================================
    # print("\n" + "="*60)
    # print("👀 来看几个你的模型亲自写的句子吧！(随机抽样)")
    # print("="*60)
    
    # # 从已经处理完的视频 id 中随机抽 5 个
    # sample_vids = random.sample(list(res.keys()), min(5, len(res)))
    # for vid in sample_vids:
    #     print(f"🎬 视频 ID: {vid}")
    #     print(f"🤖 你的模型生成: \033[92m{res[vid][0]}\033[0m") # \033[92m 是让终端输出绿色字体的魔法代码
    #     print(f"👨‍🏫 人类真实答案 (列举3个):")
    #     for gt in gts[vid][:3]:                   # 最多只打印 3 个人类写的参考答案
    #         print(f"   - {gt}")
    #     print("-" * 60)
    # # =====================================================

    # # =====================================================
    # # 定量分析：计算客观评估指标评分
    # # =====================================================
    # print("\nEvaluating metrics...")
    # try:
    #     # 定义要计算的指标列表：Bleu-1 到 Bleu-4，ROUGE_L，和 CIDEr
    #     scorers = [
    #         (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]), 
    #         (Rouge(), "ROUGE_L"), 
    #         (Cider(), "CIDEr")
    #     ]
    #     # 遍历指标工具进行打分
    #     for scorer, method in scorers:
    #         score, scores = scorer.compute_score(gts, res) # 传入真实字典 gts 和预测字典 res
    #         if type(method) == list:              # 如果是 Bleu，它会一次性返回 4 个维度的分数列表
    #             for sc, m in zip(score, method):
    #                 print(f"{m}: {sc * 100:.2f}") # 乘以 100 化为百分制并保留两位小数打印
    #         else:
    #             print(f"{method}: {score * 100:.2f}")
    # except Exception as e:
    #     print("Metric evaluation failed.")
    #     raise e    # 抓取可能出现的报错
    
        # =====================================================
    # 定量分析：计算客观评估指标评分
    # 🔥 修复：强制让 gts 和 res 的视频 ID 完全一致
    # =====================================================
    print("\nEvaluating metrics...")

    # ====================== 修复代码开始 ======================
    # 取两边都存在的共同视频 ID
    common_vids = set(gts.keys()) & set(res.keys())

    # 过滤出共同的字典，保证 key 完全一致
    gts_filtered = {vid: gts[vid] for vid in common_vids}
    res_filtered = {vid: res[vid] for vid in common_vids}
    # ====================== 修复代码结束 ======================

    try:
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        # 🔥 把 gts / res 换成过滤后的 gts_filtered / res_filtered
        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts_filtered, res_filtered)
            if type(method) == list:
                for sc, m in zip(score, method):
                    print(f"{m}: {sc * 100:.2f}")
            else:
                print(f"{method}: {score * 100:.2f}")
    except Exception as e:
        print("Metric evaluation failed.", e)
        raise e

if __name__ == "__main__":
    test()                                        # 运行测试脚本