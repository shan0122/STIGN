# STGIN Teacher Behavior

基于论文 **Spatio-temporal graph interaction networks for teacher behavior description in classroom scene** 的 PyTorch 复现工程。

## 项目结构

- `data/`
  - `dataset.py`：TBD/MSVD/MSR-VTT 数据集加载（使用预处理后 `.pt` 视频张量）
  - `preprocessing.py`：26 帧均匀采样、文本预处理、词表、检测后处理、C3D片段抽取
- `models/`
  - `tsi_module.py`：TSI（YOLOv5 检测 + RoIAlign + 图构建 + GCN）
  - `ttc_module.py`：TTC（ResNet-101 + C3D + 时序注意力）
  - `description_gen.py`：语义概念检测 + 6 层 Transformer 解码 + 束搜索
  - `stgin.py`：完整 STGIN 组装
- `utils/`
  - `detector.py`：YOLOv5 检测封装
  - `feature_extract.py`：ResNet-101/C3D 特征提取封装
  - `metrics.py`：BLEU/METEOR/CIDEr/ROUGE_L 评估
  - `loss_functions.py`：加权交叉熵（`β=0.4`）
  - `graph_ops.py`：相似度、距离、图构建、GCN 聚合
- 根目录脚本
  - `config.py`：统一超参数
  - `train.py`：训练脚本（Adam，默认 `lr=1e-6`，`batch_size=64`，支持 DDP）
  - `inference.py`：推理脚本（束搜索 `k=6`）
  - `evaluate.py`：评估脚本

## 数据格式

`annotation.json` 示例：

```json
[
  {
    "video_id": "xxx",
    "captions": ["teacher writes on board", "..."]
  }
]
```

视频张量目录需包含 `video_id.pt`，shape 为 `[T, C, H, W]`（RGB）。

## 训练

```bash
python train.py \
  --dataset TBD \
  --annotations /path/to/annotation.json \
  --video-tensors /path/to/video_tensors \
  --epochs 20
```

## 推理

```bash
python inference.py \
  --dataset TBD \
  --annotations /path/to/annotation.json \
  --video-tensors /path/to/video_tensors \
  --checkpoint /path/to/stgin_epoch_20.pt \
  --beam-size 6
```

## 评估

```bash
python evaluate.py --references refs.json --predictions preds.json
```

## 关键实现细节映射

- **TSI**
  - 保留 `conf>0.5` 且 NMS 后 Top-5 人体框
  - `sim(i,j)=cos(f_i,f_j)`，`dist(i,j)=||p_i-p_j||_2`
  - `w_ij = sim(i,j) * [dist(i,j)<threshold]`
  - GCN 使用邻居均值聚合 + ReLU，最终节点均值池化得到帧级 `v^s`
- **TTC**
  - 每帧选择最高置信度教师目标
  - ResNet-101 外观 2048 + C3D 运动 4096，拼接为 6144 引导特征
  - 多头注意力完成跨帧时序融合，输出 `v^a`
- **描述生成**
  - 语义标签维度：TBD=300、MSVD=300、MSR-VTT=400
  - Transformer 解码器：6 层、10 头、`d_model=1024`
  - 束搜索：`k=6`，最大长度 `26`

## 测试

```bash
python -m unittest discover -s tests -v
```