import json
import collections
import os
import yaml
import random
import jieba
import re

def prepare_teacher_dataset(config_path="configs/config_teacher.yaml"):
    # 1. 加载配置
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    raw_json_path = config['data']['raw_annotation']  # data/Teacher/raw_teacher.json
    output_dir = config['data']['processed_dir']      # data/Teacher/
    vocab_path = config['data']['vocab_path']         # data/Teacher/vocab.json
    max_seq_len = config['data'].get('max_seq_len', 26)
    num_concepts = config['model'].get('num_semantic_concepts', 100)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(raw_json_path):
        raise FileNotFoundError(f"找不到原始标注文件: {raw_json_path}。请先运行 step0_convert_format.py")
        
    with open(raw_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    # 2. 动态随机划分数据集 (80% Train, 10% Val, 10% Test)
    print(f"总视频数量: {len(raw_data)}")
    random.seed(42) # 固定随机种子，保证每次划分一致
    random.shuffle(raw_data)
    
    total = len(raw_data)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    
    train_data = raw_data[:train_end]
    val_data = raw_data[train_end:val_end]
    test_data = raw_data[val_end:]
    
    print(f"划分结果 -> 训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")
    
    # 3. 使用 jieba 中文分词构建词表 (Vocab)
    # 定义常见中文标点符号，用于清洗
    chinese_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    
    counter = collections.Counter()
    for item in train_data:
        for cap in item['captions']:
            # 去除标点符号和首尾空格
            clean_cap = re.sub(f"[{chinese_punctuation}]", "", cap).strip()
            # 使用结巴分词
            tokens = [word for word in jieba.lcut(clean_cap) if word.strip()]
            counter.update(tokens)
            
    # 建立词表映射
    vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
    idx = 4
    for word, count in counter.items():
        if count >= 2: # 过滤掉只出现过1次的生僻词
            vocab[word] = idx
            idx += 1
            
    print(f"中文字典 (Vocab) 大小: {len(vocab)}")
    
    # 4. 提取用于 SCD 模块的语义概念词 (Semantic Concepts)
    # 自定义中文停用词表
    stopwords = set(['了', '的', '在', '是', '着', '和', '与', '进行', '向', '边', '上', '下', '前', '后', '为', '正', '把', '被'])
    semantic_concepts = [w for w, c in counter.most_common() if w not in stopwords and c >= 2]
    # 取前 num_concepts 个作为多标签分类的目标
    semantic_concepts = semantic_concepts[:num_concepts]
    concept2idx = {word: i for i, word in enumerate(semantic_concepts)}
    
    print(f"提取的高频语义概念数量: {len(concept2idx)}")

    # 5. 处理并保存数据
    def process_split(data_split, split_name):
        processed = []
        for item in data_split:
            vid = item['video_id']
            tokens_list = []
            # 初始化全零的语义标签向量
            semantic_labels = [0.0] * num_concepts
            
            for cap in item['captions']:
                clean_cap = re.sub(f"[{chinese_punctuation}]", "", cap).strip()
                tokens = [word for word in jieba.lcut(clean_cap) if word.strip()]
                
                # 截断超长句子
                if len(tokens) > max_seq_len:
                    tokens = tokens[:max_seq_len]
                
                # 转换为数字 ID
                token_ids = [vocab.get(w, vocab['<UNK>']) for w in tokens]
                token_ids = [vocab['<BOS>']] + token_ids + [vocab['<EOS>']]
                tokens_list.append(token_ids)
                
                # 生成 SCD 模块的 Multi-hot 标签
                for w in tokens:
                    if w in concept2idx:
                        semantic_labels[concept2idx[w]] = 1.0
                        
            # 将原始的 action_labels 行为类别一同保存
            processed.append({
                'video_id': vid, 
                'video_path': item.get('video_path', f"{vid}.mp4"),
                'action_labels': item.get('action_labels', []), # 完美保留你的行为标签
                'tokens_list': tokens_list, 
                'semantic_labels': semantic_labels
            })
            
        # 注意这里文件名为 teacher_train.json 等
        save_path = os.path.join(output_dir, f'teacher_{split_name}.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)
            
    # 执行划分数据的保存
    process_split(train_data, 'train')
    process_split(val_data, 'val')
    process_split(test_data, 'test')
    
    # 保存词典
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
        
    print("✅ 教师数据集预处理完成！")

if __name__ == "__main__":
    prepare_teacher_dataset()