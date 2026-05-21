# STGIN

## 1. Setup
```bash
pip install -r requirements.txt
```

## 2. Prepare Data
Put your MSVD data into `data/raw_msvd.json`, then run:
```bash
python dataset/prepare_msvd.py
```

## 3. Train
Uses Gradient Accumulation (BS=16 * 4) and Apple MPS acceleration.
```bash
python train.py
```

## 4. Test
Generates sentences and evaluates BLEU, METEOR, ROUGE_L, and CIDEr.
```bash
python test.py
```
