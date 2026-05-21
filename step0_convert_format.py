import json
import os
import urllib.parse

def convert_dataset():
    input_json_path = "data/Teacher/label_studio_export.json" 
    output_json_path = "data/Teacher/raw_teacher.json"
    
    with open(input_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    cleaned_data = []
    
    for item in raw_data:
        # ====== 1. 解析视频路径和 ID ======
        raw_video_path = item['data']['video']
        clean_path = urllib.parse.unquote(raw_video_path).replace('\\', '/')
        file_name = clean_path.split('videos/')[-1] if 'videos/' in clean_path else clean_path.split('/')[-1]
        video_id = file_name.replace('/', '_').replace('.mp4', '').replace('.avi', '')
        
        captions = []
        action_classes = []  # 新增：用于存放行为分类标签
        
        # ====== 2. 提取文本描述和行为分类 ======
        for anno in item.get('annotations', []):
            for res in anno.get('result', []):
                # 提取文本描述 (Caption)
                if res.get('type') == 'textarea':
                    text_list = res.get('value', {}).get('text', [])
                    if text_list:
                        captions.append(text_list[0].strip())
                
                # 新增：提取行为分类标签 (Choices)
                elif res.get('type') == 'choices':
                    choices_list = res.get('value', {}).get('choices', [])
                    if choices_list:
                        # 有时候会有多选，把它们都加入列表
                        action_classes.extend(choices_list)
        
        if not captions:
            continue
            
        # 去重操作（因为如果有多个标注员，同一个类别可能会被标多次）
        action_classes = list(set(action_classes))
        
        # ====== 3. 组装格式 ======
        cleaned_data.append({
            "video_id": video_id,
            "video_path": file_name,
            "action_labels": action_classes,  # 把标签也保存进 JSON
            "captions": captions
        })
        
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 成功转换了 {len(cleaned_data)} 个视频的数据！")
    if cleaned_data:
        print("\n👀 数据格式预览：")
        print(json.dumps(cleaned_data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    convert_dataset()