import json
import jieba
from collections import Counter

# ========== 配置（改成你的路径） ==========
INPUT_TRAIN_JSON = "./data/TeacherDataset/tbd_train.json"
OUTPUT_VOCAB_JSON = "./data/TeacherDataset/vocab_optimized.json"
MIN_WORD_COUNT = 2  # 低频词汇过滤阈值
# ==========================================

# 1. 定义需要过滤的停用词和标点（教育场景专属）
STOPWORDS = {
    # 通用停用词
    "的", "在", "和", "或", "与", "向", "上", "下", "左", "右", "前", "后", 
    "旁", "中", "为", "了", "是", "有", "能", "会", "可", "把", "被", "让", 
    "给", "对", "跟", "同", "及", "也", "还", "都", "只", "才", "就", "又",
    # 标点符号
    "，", "。", "！", "？", "；", "：", "、", "（", "）", "【", "】", "《", "》", 
    "“", "”", "‘", "’", ".", ",", "!", "?", ";", ":", "(", ")", "[", "]"
}

# 2. 定义教育核心词汇白名单（确保关键词汇不被过滤）
EDU_CORE_WORDS = {
    "教师", "学生", "教学", "多媒体", "大屏幕", "屏幕", "讲台", "讲解", "讲授", 
    "结合", "区域", "板书", "课件", "互动", "提问", "巡课", "教案", "黑板"
}

def main():
    # 读取训练集标注
    with open(INPUT_TRAIN_JSON, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    # 收集所有描述文本
    all_text = []
    for sample in train_data:
        for caption in sample["captions"]:
            if caption.strip():
                all_text.append(caption.strip())
    
    print(f"✅ 共收集 {len(all_text)} 条描述文本")
    
    # 分词 + 过滤停用词/标点
    word_list = []
    for text in all_text:
        words = jieba.lcut(text)
        # 过滤规则：
        # 1. 非停用词/标点  2. 非空  3. 非纯数字  4. 长度≥1
        valid_words = [
            word for word in words 
            if word not in STOPWORDS 
            and word.strip() 
            and not word.isdigit() 
            and len(word) >= 1
        ]
        word_list.extend(valid_words)
    
    # 统计词频
    word_counter = Counter(word_list)
    print(f"✅ 分词+过滤停用词后，原始词汇数：{len(word_counter)}")
    
    # 过滤低频词汇 + 保留核心教育词汇（即使低频也保留）
    filtered_words = []
    for word, count in word_counter.items():
        # 保留条件：词频≥阈值 或 属于教育核心词汇
        if count >= MIN_WORD_COUNT or word in EDU_CORE_WORDS:
            filtered_words.append(word)
    
    # 按词频降序排序
    filtered_words.sort(key=lambda x: word_counter[x], reverse=True)
    print(f"✅ 过滤低频词汇后，核心词汇数：{len(filtered_words)}")
    
    # 构建词汇表（保留特殊标记）
    special_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
    vocab = {
        "word2idx": {},
        "idx2word": {}
    }
    
    # 添加特殊标记
    for idx, token in enumerate(special_tokens):
        vocab["word2idx"][token] = idx
        vocab["idx2word"][idx] = token
    
    # 添加过滤后的核心教育词汇
    start_idx = len(special_tokens)
    for idx, word in enumerate(filtered_words):
        vocab["word2idx"][word] = start_idx + idx
        vocab["idx2word"][start_idx + idx] = word
    
    # 保存优化后的词汇表
    with open(OUTPUT_VOCAB_JSON, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    # 输出统计信息
    print(f"\n🎉 优化版教师数据集词汇表生成完成！")
    print(f"📁 保存路径：{OUTPUT_VOCAB_JSON}")
    print(f"📊 词汇表统计：")
    print(f"   - 特殊标记：{len(special_tokens)} 个")
    print(f"   - 核心教育词汇：{len(filtered_words)} 个")
    print(f"   - 词汇表总大小：{len(vocab['word2idx'])} 个")
    
    # 输出前20个高频核心教育词汇
    print(f"\n🔥 前20个高频教育核心词汇：")
    top_20_words = []
    for word, count in word_counter.most_common():
        if word in filtered_words and word not in STOPWORDS:
            top_20_words.append((word, count))
            if len(top_20_words) >= 20:
                break
    
    for i, (word, count) in enumerate(top_20_words, 1):
        print(f"   {i:2d}. {word:<8} 出现次数：{count}")

if __name__ == "__main__":
    main()