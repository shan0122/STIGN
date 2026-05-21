import torch
import yaml
import json
import os
import random
import cv2
import textwrap
import glob
import matplotlib.pyplot as plt
from models.stgin import STGIN
from dataset.dataloader import get_dataloader

def generate_causal_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

def beam_search(model, roi_feats, coords, global_feats, config, device):
    """自回归贪心搜索"""
    model.eval()
    bos, eos = 1, 2 
    seq = [bos]
    with torch.no_grad():
        for _ in range(30):
            tgt_tensor = torch.tensor([seq]).to(device)
            tgt_mask = generate_causal_mask(tgt_tensor.size(1)).to(device)
            logits, _ = model(roi_feats, coords, global_feats, tgt_tensor, tgt_mask)
            next_word = logits[0, -1, :].argmax(dim=-1).item()
            if next_word == eos: break
            seq.append(next_word)
    return seq

def extract_video_frames(video_path, num_frames=8):
    """从原视频中均匀抽取指定数量的帧"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total_frames > 0:
        step = max(total_frames // num_frames, 1)
        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(i * step, total_frames - 1))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
    cap.release()
    return frames

def main():
    with open("configs/config_teacher.yaml", "r") as f:
        config = yaml.safe_load(f)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    with open(config['data']['vocab_path'], 'r') as f:
        vocab = json.load(f)
    idx2word = {idx: word for word, idx in vocab.items()}
    
    model = STGIN(
        vocab_size=len(vocab), feature_dim=config['model']['feature_dim'],
        num_semantic_concepts=config['model']['num_semantic_concepts'],
        num_layers=config['model']['transformer_layers'],
        num_heads=config['model']['transformer_heads'], mu=config['model']['mu']
    ).to(device)
    model.load_state_dict(torch.load("stgin_teacher_best.pth", map_location=device))
    model.eval()
    print("✅ Model loaded successfully!")

    output_dir = "vis_outputs"
    os.makedirs(output_dir, exist_ok=True)

    test_loader = get_dataloader(config, split='test')

    with open(os.path.join(config['data']['processed_dir'], 'teacher_test.json'), 'r') as f:
        test_annotations = json.load(f)
    
    gts = {}
    for item in test_annotations:
        if 'captions' in item:
            gts[item['video_id']] = item['captions']
        else:
            gts[item['video_id']] = [" ".join([idx2word.get(idx, '<UNK>') for idx in seq if idx not in (0, 1, 2)]) for seq in item['tokens_list']]

    print("🎨 Generating visual results...")
    
    # ================= 核心修改：先随机抽取 =================
    all_vids = list(gts.keys())
    # 随机打乱所有视频 ID
    random.shuffle(all_vids)
    
    # 提取所有 batch 数据放入字典，方便按 ID 快速查找
    print("⏳ Caching test features...")
    batch_cache = {}
    for batch in test_loader:
        vid_id = batch['video_id'][0]
        if vid_id not in batch_cache:
            batch_cache[vid_id] = batch

    saved_count = 0
    # 遍历打乱后的视频 ID，直到成功画出 5 张图
    for vid_id in all_vids:
        if saved_count >= 5: 
            break
            
        # 如果这个随机抽到的视频在 loader 里没有特征，跳过
        if vid_id not in batch_cache:
            continue

        video_pattern = os.path.join("data", "Teacher", "raw_videos",  vid_id + '.*')
        matches = glob.glob(video_pattern)
        if not matches:
            continue
            
        video_path = matches[0]
        
        batch = batch_cache[vid_id]
        roi = batch['roi_feats'][0:1].to(device)
        coords = batch['coords'][0:1].to(device)
        glb = batch['global_feats'][0:1].to(device)
        
        best_seq = beam_search(model, roi, coords, glb, config, device)
        gen_words = " ".join([idx2word.get(idx, '') for idx in best_seq if idx not in (0, 1, 2)])
        gt_words = gts[vid_id][0] 
        
        frames = extract_video_frames(video_path, num_frames=8)
        if len(frames) < 8:
            continue
            
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"Video ID: {vid_id}", fontsize=16, fontweight='bold', y=0.98)
        
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(frames[i])
            ax.axis('off')
            
        wrapped_gen = "\n".join(textwrap.wrap(f"[AI Prediction]  {gen_words}", width=80))
        wrapped_gt = "\n".join(textwrap.wrap(f"[Ground Truth]   {gt_words}", width=80))
        
        text_str = f"{wrapped_gen}\n\n{wrapped_gt}"
        
        fig.text(0.5, 0.02, text_str, ha='center', va='bottom', fontsize=16, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
        
        plt.subplots_adjust(bottom=0.2) 
        
        save_path = os.path.join(output_dir, f"vis_{vid_id}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📸 Saved visualization: {save_path}")
        saved_count += 1

    print(f"\n🎉 Done! Check the '{output_dir}' folder to see 5 random unique results!")

if __name__ == "__main__":
    main()