# API版 分散型継続事前学習実験 (Distributed CPT-AES via API)

NFSを使わず、APIサーバー経由で実験を分散実行するシステムです。
全てのPC（サーバー・クライアント）はSSH経由でアクセスする想定です。

## 構成

```
distributed_exp_api/
├── server/          # APIサーバー（1台で実行）
│   ├── main.py      # FastAPI サーバー
│   ├── setup.sh     # セットアップスクリプト
│   ├── run_background.sh  # バックグラウンド起動（SSH用）
│   ├── stop.sh      # サーバー停止
│   └── data/        # データ保存場所（自動生成）
│       ├── asap/    # ASAPデータセット
│       ├── tasks/   # タスク状態
│       ├── checkpoints/  # モデルチェックポイント
│       └── results/      # 実験結果
│
├── client/          # ワーカークライアント（各GPUマシンで実行）
│   ├── worker.py    # ワーカースクリプト
│   ├── api_client.py # API通信クライアント
│   ├── setup.sh     # セットアップスクリプト
│   ├── run_background.sh  # バックグラウンド起動（SSH用）
│   ├── stop.sh      # ワーカー停止
│   └── src/         # 実験コード
│
└── shared/          # 共有モデル定義
    └── models.py    # Pydanticモデル
```

## 実験内容

- **8 prompts** (ASAP 1-8) × **3 models** (Llama, Qwen, Mistral) = **24実験**
- 各実験: 30エポックの継続事前学習 + ゼロショット採点（Greedy Decoding）
- 1実験あたり約1-2時間（GPUによる）

## クイックスタート

### 1. サーバーPC（SSH接続）

```bash
# uvのインストール（まだの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# リポジトリをクローン
git clone https://github.com/Kitatai/distributed-cpt-aes-api.git
cd distributed-cpt-aes-api/server

# セットアップ
bash setup.sh

# ASAPデータをコピー
cp /path/to/training_set_rel3.tsv data/asap/

# サーバーをバックグラウンドで起動（SSH切断後も継続）
bash run_background.sh

# タスク初期化（初回のみ）
curl -X POST "http://localhost:8000/tasks/init"

# ログ確認
tail -f server.log
```

### 2. クライアントPC（SSH接続、各GPUマシン）

**sudoは不要です**

```bash
# uvのインストール（まだの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# リポジトリをクローン
git clone https://github.com/Kitatai/distributed-cpt-aes-api.git
cd distributed-cpt-aes-api/client

# セットアップ
bash setup.sh

# Hugging Faceログイン（ゲート付きモデル用）
uv run huggingface-cli login

# ワーカーをバックグラウンドで起動（SSH切断後も継続）
bash run_background.sh --server http://SERVER_IP:8000

# ログ確認
tail -f worker.log
```

## コマンドリファレンス

### サーバー

```bash
# バックグラウンドで起動（推奨、SSH切断後も継続）
bash run_background.sh

# 別ポートで起動
bash run_background.sh 8080

# サーバー停止
bash stop.sh

# ログ確認
tail -f server.log

# フォアグラウンドで起動（デバッグ用）
bash run.sh
```

### クライアント

```bash
# バックグラウンドで起動（推奨、SSH切断後も継続）
bash run_background.sh --server http://SERVER_IP:8000

# ワーカー停止
bash stop.sh

# ログ確認
tail -f worker.log

# フォアグラウンドで起動（デバッグ用）
bash run.sh --server http://SERVER_IP:8000

# 1タスクだけ実行して終了
bash run.sh --server http://SERVER_IP:8000 --single

# カスタムワーカーID
bash run_background.sh --server http://SERVER_IP:8000 --worker-id mypc01
```

### 進捗確認（サーバー側）

```bash
# マトリクス形式（P=Pending, R=Running, C=Completed, F=Failed）
curl "http://localhost:8000/tasks/matrix" | python -m json.tool

# 全タスクの詳細
curl "http://localhost:8000/tasks" | python -m json.tool

# CLIツールで確認
uv run python status.py --matrix
```

### タスク管理

```bash
# タスク初期化（初回のみ）
curl -X POST "http://localhost:8000/tasks/init"

# 全タスク強制リセット
curl -X POST "http://localhost:8000/tasks/init?force=true"

# 特定タスクのリセット
curl -X POST "http://localhost:8000/tasks/prompt1_llama/reset"
```

## データの流れ

```
クライアント                          サーバー
    │                                    │
    ├─── タスク取得リクエスト ──────────→│
    │←── タスク情報 + ASAPデータ ────────┤
    │                                    │
    │    [ローカルでGPU学習]             │
    │                                    │
    ├─── checkpoint + results ──────────→│ (各エポック後)
    │                                    │
    ├─── タスク完了通知 ────────────────→│
    │                                    │
```

## Resume機能

実験が途中で中断しても、自動的に続きから再開されます：

1. ワーカーがタスクを取得
2. サーバーから最後に完了したエポックを確認
3. 必要に応じてチェックポイントをダウンロード
4. 続きのエポックから学習再開

## 設定変更

`client/src/config.py` で以下を調整できます：

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `max_seq_len` | 2048 | 最大シーケンス長 |
| `batch_size` | 1 | バッチサイズ |
| `grad_accum_steps` | 1 | 勾配累積ステップ |
| `gradient_checkpointing` | False | メモリ節約（有効にすると遅くなる） |
| `logit_extraction.enabled` | False | 期待値計算（無効で高速化） |

## トラブルシューティング

### サーバーに接続できない

```bash
# サーバー側でファイアウォール確認
sudo ufw status

# ポート開放（必要なら）
sudo ufw allow 8000
```

### GPU メモリエラー

`client/src/config.py` で以下を調整：
- `max_seq_len` を下げる（512, 256など）
- `gradient_checkpointing = True` に変更

### Hugging Face認証エラー

```bash
uv run huggingface-cli login
```

### Flash Attentionのビルドエラー

オプションなので無視しても問題ありません（少し遅くなるだけ）。

### SSH切断後にプロセスが終了する

`run_background.sh` を使用してください：
```bash
bash run_background.sh --server http://SERVER_IP:8000
```

## ディレクトリ構造

### サーバー側（実験結果の保存場所）
```
server/data/
├── asap/training_set_rel3.tsv    # ASAPデータセット
├── tasks/*.json                   # タスク状態ファイル
├── checkpoints/{task_id}/         # チェックポイント
│   └── epoch_{N}/adapter.zip
└── results/{task_id}/             # 実験結果
    ├── metrics_epoch_{N}.json
    ├── predictions_epoch_{N}.csv
    └── summary.json
```

### クライアント側（一時作業ディレクトリ）
```
~/.cpt-aes-worker/
├── asap/training_set_rel3.tsv    # ダウンロードしたデータ
└── work/{task_id}/               # 作業ディレクトリ
    ├── checkpoints/
    ├── results/
    └── splits/
```

## 注意事項

- サーバーは実験中は起動し続けてください
- クライアントは途中で停止しても、再起動すれば続きから実行されます
- 最終結果はすべてサーバーの `data/results/` に保存されます
- `run_background.sh` を使えばSSH切断後もプロセスは継続します
