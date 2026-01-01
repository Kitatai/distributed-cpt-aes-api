# Few-shot v2 実験ガイド

## 概要

この実験では、継続事前学習（CPT）で学習したLoRAアダプタを用いて、few-shot採点を行います。

### 従来のアプローチとの違い

| 項目 | 従来 (few-shot v1) | 今回 (few-shot v2) |
|------|-------------------|-------------------|
| Dev分割 | 10件中5件をshot | 10件中3件をshot |
| エポック選択 | zero-shot分析の結果を使用 | 7件でMSE計算して選択 |
| 評価対象 | 残り全エッセイ | 残りの10%をサンプル |

### 実験フロー

1. 10サンプルを3 shot + 7 devに分割
2. エポック0-30で7 devをfew-shot採点 → MSE算出
3. MSEが最小のエポックを選択
4. 最良エポックでテストセット（残りの10%）を評価
5. QWKと順位相関を報告

## データ構造

```
server/data/backup_zeroshot_v1/
├── checkpoints/              # LoRAアダプタ (checkpoints.zipから解凍済み)
│   ├── prompt1_llama8b/
│   │   ├── epoch_1/adapter.zip
│   │   ├── epoch_2/adapter.zip
│   │   └── ...
│   └── ...
├── sample_patterns.json      # 50パターン × 10サンプルID
├── tasks_fewshot_v2/         # 生成されたタスク
└── results_fewshot_v2/       # 実験結果
```

## クイックスタート

### 1. セットアップ（サーバー側）

```bash
cd distributed_exp_api/server

# タスク生成（8プロンプト × 1パターン × llama8b）
./run_fewshot_v2_setup.sh

# カスタムオプション
./run_fewshot_v2_setup.sh "1,2,3" 1 llama8b  # プロンプト1-3のみ
./run_fewshot_v2_setup.sh "1" 50 llama8b     # プロンプト1で50パターン
```

### 2. 実験実行（クライアント側）

```bash
cd distributed_exp_api/client

# 全タスク実行（バックグラウンド）
./run_fewshot_v2_worker.sh

# 特定タスクのみ
./run_fewshot_v2_worker.sh fewshot_v2_prompt1_llama8b_p0

# 特定プロンプトのみ
./run_fewshot_v2_worker.sh "" 1
```

### 3. ログ監視

```bash
# 最新ログを表示
tail -f client/logs/fewshot_v2_*.log

# ワーカー停止
kill <PID>
```

## 詳細コマンド

### タスク生成オプション

```bash
python generate_fewshot_v2_tasks.py \
    --prompts 1,2,3,4,5,6,7,8 \  # 対象プロンプト
    --patterns 1 \               # パターン数 (1-50)
    --model llama8b \            # モデル (llama8b, llama3b, mistral)
    --n-shot 3 \                 # few-shot数
    --test-ratio 0.1             # テストセットのサンプル比率
```

### ワーカーオプション

```bash
python worker_fewshot_v2.py \
    --data-dir ../server/data/backup_zeroshot_v1 \
    --task-id fewshot_v2_prompt1_llama8b_p0  # 特定タスク
    # or
    --prompt 1  # 特定プロンプトの全タスク
```

## 出力ファイル

各タスクの結果は `results_fewshot_v2/<task_id>/` に保存されます：

```
results_fewshot_v2/fewshot_v2_prompt1_llama8b_p0/
├── summary.json                  # 最終結果サマリー
├── test_predictions_epoch0.csv   # ベースライン(epoch 0)の予測
├── test_predictions_best.csv     # 最良エポックの予測
├── epoch_mse_curve.csv           # エポック別MSE
└── dev_epoch_*.json              # 各エポックのdev評価結果
```

### summary.json の内容

```json
{
  "task_id": "fewshot_v2_prompt1_llama8b_p0",
  "prompt_id": 1,
  "epoch_selection": {
    "best_epoch": 15,
    "best_mse": 2.5714
  },
  "test_baseline": {
    "epoch": 0,
    "qwk": 0.35,
    "spearman": 0.42,
    "n_samples": 178
  },
  "test_best": {
    "epoch": 15,
    "qwk": 0.45,
    "spearman": 0.52,
    "n_samples": 178
  },
  "improvement": {
    "qwk": 0.10,
    "spearman": 0.10,
    "mse": 1.5
  }
}
```

## 計算時間の目安

- 1エポックあたり: 7 devサンプル × 約2秒 = 約15秒
- 全エポック (0-30): 約8分
- テスト評価 (epoch 0 + best): 約180サンプル × 2回 × 約2秒 = 約12分
- **1タスク合計: 約20分**
- **8プロンプト × 1パターン: 約2.5時間**

## トラブルシューティング

### Checkpointsが見つからない

```bash
cd server/data
unzip -q checkpoints.zip -d backup_zeroshot_v1/
```

### CUDA Out of Memory

1タスクずつ実行するか、より小さいモデル (llama3b) を使用

### タスクが完了済みになる

結果を削除してから再実行：
```bash
rm -rf server/data/backup_zeroshot_v1/results_fewshot_v2/<task_id>
```

## 注意事項

- `backup_zeroshot_v1/` のデータは zero-shot v1 実験の結果なので削除しないこと
- 現在の `checkpoints/` は base model 訓練の結果（別実験）
- この実験では `backup_zeroshot_v1/checkpoints/` のアダプタを使用
