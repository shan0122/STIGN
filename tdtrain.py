import json
import random
import os

# ========== 配置（改成你自己的路径） ==========
INPUT_JSON_PATH = "TeacherDataset.json"  # 你导出的原始JSON
OUTPUT_DIR = "./data/TeacherDataset"   # 输出目录
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
# =============================================

def main():
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed_samples = []
    for item in raw_data:
        # 视频路径和ID
        video_path = item["data"]["video"]
        video_path = video_path.replace("\\", "/").replace("?d=", "")
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        captions = []
        behavior_types = []

        for ann in item["annotations"]:
            for res in ann["result"]:
                value = res["value"]

                # 1. 提取文本描述
                if "text" in value:
                    text_content = value["text"]
                    if isinstance(text_content, list):
                        if len(text_content) > 0:
                            text = str(text_content[0]).strip()
                    else:
                        text = str(text_content).strip()
                    if text:
                        captions.append(text)

                # 2. 修复：提取行为类型（choices/labels 字段）
                if "choices" in value and value["choices"]:
                    # 你的标注里用的是 choices 字段
                    behavior_types.extend(value["choices"])
                elif "labels" in value and value["labels"]:
                    # 兼容 labels 字段
                    if isinstance(value["labels"], list):
                        behavior_types.extend(value["labels"])
                    else:
                        behavior_types.append(value["labels"])

        # 去重并过滤空字符串
        behavior_types = list(set([bt.strip() for bt in behavior_types if bt.strip()]))

        if captions or behavior_types:
            processed_samples.append({
                "video_id": video_id,
                "video_path": video_path,
                "captions": captions,
                "behavior_types": behavior_types
            })

    print(f"✅ 处理完成，有效样本数：{len(processed_samples)}")

    # 统计行为类型分布
    all_behaviors = {}
    for sample in processed_samples:
        for bt in sample["behavior_types"]:
            all_behaviors[bt] = all_behaviors.get(bt, 0) + 1

    print("\n📊 行为类型分布：")
    if all_behaviors:
        for bt, count in all_behaviors.items():
            print(f"   {bt}: {count} 个样本")
    else:
        print("   ⚠️  仍未提取到行为类型，请检查标注文件格式")

    # 划分训练/验证/测试集
    random.shuffle(processed_samples)
    n = len(processed_samples)
    train_end = int(n * SPLIT_RATIOS["train"])
    val_end = train_end + int(n * SPLIT_RATIOS["val"])

    splits = {
        "train": processed_samples[:train_end],
        "val": processed_samples[train_end:val_end],
        "test": processed_samples[val_end:]
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split_name, samples in splits.items():
        out_path = os.path.join(OUTPUT_DIR, f"tbd_{split_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 已保存 {split_name} 集：{out_path}")
        print(f"   样本数：{len(samples)}")
        split_behaviors = {}
        for s in samples:
            for bt in s["behavior_types"]:
                split_behaviors[bt] = split_behaviors.get(bt, 0) + 1
        print(f"   行为类型覆盖：{list(split_behaviors.keys())}")

if __name__ == "__main__":
    main()