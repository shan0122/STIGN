import json                                  # 导入 json 模块，用于读写 .json 格式的标注文件
import collections                           # 导入 collections，主要使用里面的 Counter（计数器）来统计单词词频
import string                                # 导入 string，主要用于获取所有的标点符号集合
import os                                    # 导入 os，用于处理文件路径和创建文件夹
import yaml                                  # 导入 yaml，用于读取 YAML 格式的配置文件

# ==========================================
# 主函数：准备 MSVD 数据集
# ==========================================
def prepare_msvd_dataset(config_path="configs/config_msvd.yaml"):
    # 1. 打开并读取 yaml 配置文件
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)           # 将 yaml 解析为 Python 字典格式
        
    raw_json_path = config['data']['raw_annotation']  # 获取原始 MSVD 标注文件的存放路径
    output_dir = config['data']['processed_dir']      # 获取预处理后数据的输出文件夹路径
    
    os.makedirs(output_dir, exist_ok=True)            # 创建输出文件夹，如果已经存在则不会报错 (exist_ok=True)
    
    # 2. 【兜底机制】如果你还没有下载真实的 MSVD 数据集，脚本会自动生成一个假的数据集，保证代码能往下跑跑通流程
    if not os.path.exists(raw_json_path):
        print(f"Warning: {raw_json_path} not found. Generating dummy raw data for structure testing.")
        # 生成 1970 个假视频记录，每个视频配两句假描述
        dummy_data = [{"video_id": f"vid{i}", "captions": ["a man is playing guitar", "someone is playing music"]} for i in range(1970)]
        with open(raw_json_path, 'w') as f:
            json.dump(dummy_data, f)         # 将假数据写入 raw_json_path 文件
            
    # 3. 读取原始数据（真实的或刚刚生成的假的）
    with open(raw_json_path, 'r') as f:
        raw_data = json.load(f)              # 加载后 raw_data 是一个列表，里面有很多字典
        
    # ==========================================
    # 步骤 1: 划分数据集 (Dataset Splitting)
    # 根据学术界通用标准，MSVD 数据集划分为：前 1200 个用于训练，接下来 100 个用于验证，最后 670 个用于测试
    # ==========================================
    train_data = raw_data[:1200]             
    val_data = raw_data[1200:1300]
    test_data = raw_data[1300:1970]
    
    # ==========================================
    # 步骤 2: 构建词表 (Vocabulary Building)
    # 注意：词表只能使用训练集 (train_data) 来构建，绝对不能看验证集和测试集，否则会造成数据穿越(Data Leakage)
    # ==========================================
    counter = collections.Counter()          # 初始化一个计数器对象
    for item in train_data:                  # 遍历所有的训练样本
        for cap in item['captions']:         # 遍历某个视频包含的所有句子
            # 文本清洗：
            # 1. cap.lower() 转换为全小写
            # 2. translate(...) 删除所有的标点符号
            # 3. split() 按照空格切分成单词列表
            tokens = cap.lower().translate(str.maketrans('', '', string.punctuation)).split()
            counter.update(tokens)           # 把这些单词扔进计数器里统计频率
            
    # 初始化词典，赋予 4 个特殊词汇固定索引：
    # <PAD>：占位符；<BOS>：句子开头(Begin of Sentence)；<EOS>：句子结尾(End)；<UNK>：词典里没有的未知词(Unknown)
    vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
    idx = 4                                  # 常规单词从索引 4 开始分配
    
    for word, count in counter.items():      # 遍历统计出的所有单词及其频率
        if count >= 2:                       # 【频率过滤】过滤掉只出现过 1 次的生僻词（拼写错误等），以缩小词表大小
            vocab[word] = idx                # 给单词分配数字索引
            idx += 1
            
    print(f"MSVD Vocabulary Size: {len(vocab)}") # 打印最终词表包含的单词总数
    
    # ==========================================
    # 步骤 3: 提取用于 SCD 模块的语义概念词 (Semantic Concepts)
    # 作用：找出视频中最常出现的实体名词或动作动词，让模型做多标签分类（也就是视频里有没有这个东西）
    # ==========================================
    # 停用词表：去除掉冠词、代词、连词等没有视觉意义的词，以及太笼统的 man/woman 等词
    stopwords = set(['a', 'the', 'is', 'are', 'in', 'on', 'of', 'and', 'to', 'with', 'man', 'woman', 'boy', 'girl'])
    
    # 从 counter 里选出频率最高的词 (most_common)，过滤掉停用词和生僻词，取前 300 个作为核心语义概念
    semantic_concepts = [w for w, c in counter.most_common() if w not in stopwords and c >= 2][:300]
    
    # 建立这 300 个概念的逆向索引映射，用于后续生成 One-hot 标签
    concept2idx = {word: i for i, word in enumerate(semantic_concepts)}
    
    # ==========================================
    # 步骤 4: 处理并保存数据集 (Process and Save)
    # ==========================================
    # 定义一个内部函数，用于处理某种切分（训练集/验证集/测试集）
    def process_split(data_split, split_name):
        processed = []                       # 用于存放处理后的该子集数据
        
        for item in data_split:              # 遍历子集里的每个视频样本
            vid = item['video_id']           
            tokens_list = []                 # 存放该视频下所有句子转换后的数字索引序列
            semantic_labels = [0.0] * 300    # 初始化长度为 300 的全零列表，表示这 300 个概念还没出现
            
            for cap in item['captions']:     # 遍历句子的原始英文字符串
                # 和之前一样的文本清洗操作
                tokens = cap.lower().translate(str.maketrans('', '', string.punctuation)).split()
                
                # 如果句子长度大于 26，强制截断到 26（防止少数超长句子消耗过多显存）
                if len(tokens) > 26: tokens = tokens[:26]
                
                # 将英文单词列表转化为数字索引列表。如果单词没在词表里，就给 <UNK> 的索引 (3)
                token_ids = [vocab.get(w, vocab['<UNK>']) for w in tokens]
                
                # 在句首加上 <BOS>(1)，句尾加上 <EOS>(2)
                token_ids = [vocab['<BOS>']] + token_ids + [vocab['<EOS>']]
                tokens_list.append(token_ids)
                
                # 生成 300 维的 multi-hot 语义标签（如果这句描述里出现了前300大概念里的词，就把它对应的位置标为 1.0）
                for w in tokens:
                    if w in concept2idx:
                        semantic_labels[concept2idx[w]] = 1.0
                        
            # 将处理好的这个视频的纯数字信息存入列表
            processed.append({'video_id': vid, 'tokens_list': tokens_list, 'semantic_labels': semantic_labels})
            
        # 将处理好的子集列表，存为一个 JSON 文件（如 msvd_train.json）
        with open(os.path.join(output_dir, f'msvd_{split_name}.json'), 'w') as f:
            json.dump(processed, f)
            
    # 依次调用函数，处理并保存 train, val, test 三个部分
    process_split(train_data, 'train')
    process_split(val_data, 'val')
    process_split(test_data, 'test')
    
    # 最终保存词典本身（推理时要根据这个词典把数字翻译回英文单词）
    with open(config['data']['vocab_path'], 'w') as f:
        json.dump(vocab, f)
        
    print("Dataset preparation complete.")

if __name__ == "__main__":
    prepare_msvd_dataset()                   # 运行主函数