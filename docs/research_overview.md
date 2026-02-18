# マルチモーダルLLMを用いた音声対話状態追跡の強化学習

## 1. 研究概要
本研究は、マルチモーダル大規模言語モデル **Qwen2.5-Omni-7B** を用いて、音声入力から直接 **対話状態追跡（Dialogue State Tracking; DST）** を行うシステムを構築し、**GRPO（Group Relative Policy Optimization）** による強化学習で性能を向上させる取り組みである。

### 1.1 研究の動機

タスク指向対話システムとは，ユーザとの対話を通じて，レストラン予約などのタスクを自動で処理するシステムである．近年，スマートスピーカー等の普及により，タスク指向対話システムの音声入力への対応が求められている．

従来，音声を入力とする対話状態追跡（Dialogue State Tracking: DST）では，自動音声認識（Auto Speech Recognition: ASR）とテキストベースの DST モデルを組み合わせたパイプライン手法が一般的であった．しかし，この手法では，ASRの誤りが後段のDSTに伝播し，回復困難なカスケードエラーを引き起こすという問題があった．この課題に対し，音声をテキストに変換する中間過程を経ず，音声を直接入力として受け取って対話状態を推定するエンドツーエンド（E2E）モデルが注目を集めている．

DSTのタスク定式化には，各ターンで累積された対話状態全体を出力する「全状態出力方式（Full-State DST）」と，前ターンからの変更点のみを出力する「増分ベース方式（Incremental DST）」が存在する．増分ベース方式は，スロットの追加（set），更新（update），削除（delete）といった差分操作のみを予測するため，出力が簡潔であり，ユーザの要求の変化に焦点を当てやすく，全状態出力方式に比べ各ターンにおいてモデルが行うタスクが単純になるという利点がある．

しかし，システム自身の予測を次ターンに引き継ぐ実践的なカスケード評価（予測モード）においては，過去のターンの差分操作の誤りが連鎖的に蓄積しやすいため，極めて高い精度と安定性が要求される．

E2Eモデルの学習には，正解ラベルを与える教師ありファインチューニング（SFT）が広く用いられる．しかし，SFTのみでは，音声特有の揺れや差分操作の複雑な依存関係に対して，十分な頑健性を獲得することが難しい．特に増分ベースのDSTにおいては，生成された差分操作の正確性やフォーマットの正当性が後続ターンの状態に直結するため，モデル出力全体に対する定量的な評価指標を直接最適化する枠組みが必要不可欠である．

そこで本研究では，マルチモーダル大規模言語モデル（Qwen2.5-Omni-7B）を用いた，音声入力による増分ベースの対話状態追跡に対する新たな二段階学習フレームワークを提案する．具体的には，SFTにより基礎的なタスク遂行能力を獲得させた後，Group Relative Policy Optimization（GRPO）による強化学習を導入する  ．GRPOでは，価値関数（Critic）を用いず，同一の音声入力からサンプリングした複数の推論結果に対する相対的な報酬（転写テキストの単語誤り率，差分操作のF1スコア，フォーマットスコア等の重み付き和）に基づいてポリシーを更新する．これにより，音声からの直接的な文脈理解と増分ベースの状態更新の精度を向上させ，実用的な予測モードにおいても頑健な音声DSTの実現を目指す

### 1.2 研究目標

1. **音声直接入力によるDST**: テキストを介さず音声から直接対話状態を追跡
2. **2つのDST定式化の比較**: Incremental DST（差分操作）vs Full-State DST（全状態出力）
3. **SFT + GRPO二段階学習**: 教師あり微調整の後に強化学習で性能を向上
4. **予測モードでの実用的評価**: モデル自身の予測を次ターンに引き継ぐカスケード評価

---

## 2. モデルアーキテクチャ

### 2.1 ベースモデル

| 項目 | 値 |
|------|-----|
| モデル | Qwen2.5-Omni-7B |
| パラメータ数 | 7B |
| 入力モダリティ | テキスト + 音声 |
| 音声処理 | 統合音声エンコーダ |

### 2.2 ファインチューニング構成

| 項目 | 値 |
|------|-----|
| 手法 | LoRA (Low-Rank Adaptation) |
| LoRA Rank | 64 |
| LoRA Alpha | 128 |
| 量子化 | 4-bit NF4 + Double Quantization |
| 計算精度 | bfloat16 |
| 分散学習 | DeepSpeed ZeRO-2 |
| Attention | Flash Attention 2 |
| 勾配チェックポイント | 有効 |

---

## 3. タスク定式化

本研究では2つの対話状態追跡の定式化を実装・比較する。

### 3.1 Incremental DST（差分操作方式）

各ターンで **前ターンからの差分のみ** を予測する方式。

**入力**:
```
[Dialogue History]
User: hello is this customer service center.
System: yes, how may i help.

[Previous State]
{"restaurant":{"area":"centre"}}

[New Audio]
<audio>
```

**出力**:
```
<transcript>
System: do you have a price preference.
User: i'd like something cheap.
</transcript>
<answer>set(restaurant.pricerange=cheap)</answer>
```

**差分操作の種類**:

| 操作 | 形式 | 意味 |
|------|------|------|
| set | `set(domain.slot=value)` | 新規スロットの追加 |
| update | `update(domain.slot=value)` | 既存スロットの値更新 |
| delete | `delete(domain.slot)` | スロットの削除 |

### 3.2 Full-State DST（全状態出力方式）

各ターンで **累積された対話状態全体** をJSON形式で出力する方式。

**入力**:
```
[Dialogue History]
User: hello is this customer service center.
System: yes, how may i help.

[New Audio]
<audio>
```

**出力**:
```
<transcript>
System: do you have a price preference.
User: i'd like something cheap.
</transcript>
<answer>{"restaurant":{"pricerange":"cheap","area":"centre"}}</answer>
```

### 3.3 方式の比較

| 観点 | Incremental DST | Full-State DST |
|------|----------------|----------------|
| 出力形式 | 差分操作（set/update/delete） | 完全なJSON状態 |
| 入力に含む状態 | 前ターンの対話状態 | なし |
| 予測モードでの伝播 | 転写 + 差分操作の適用結果 | 転写のみ（状態は毎回独立生成） |
| 利点 | 出力が簡潔、変更に焦点 | 実装がシンプル、状態の蓄積エラーが起きにくい |
| 欠点 | 差分操作の適用エラーが蓄積 | 出力が冗長、ターンが進むと出力が長くなる |

---

## 4. 学習パイプライン

### 4.1 Stage 1: 教師あり微調整（SFT）

正解データで教師あり学習を行い、タスクの基本的な能力を獲得する。

**学習設定**:

| パラメータ | 値 |
|-----------|-----|
| GPU数 | 2 |
| バッチサイズ | 1 |
| 勾配蓄積 | 16ステップ |
| 学習率 | 1e-4 |
| エポック数 | 3 |
| 最大系列長 | 4096 |
| 評価間隔 | 200ステップ |
| 保存間隔 | 200ステップ |
| WandBプロジェクト | qwenomni-sft |

**学習経過** (Incremental DST):
- 初期 Loss: ~0.13–0.14
- 最終 Loss: ~0.078
- トークン精度: 97.6%
- 総ステップ数: 7,470

### 4.2 Stage 2: 強化学習（GRPO）

SFTチェックポイントを出発点として、報酬関数に基づくGRPO学習で性能をさらに向上させる。

**学習設定**:

| パラメータ | 値 |
|-----------|-----|
| 学習GPU数 | 8 |
| 推論GPU数 | 2 |
| バッチサイズ | 2 |
| 勾配蓄積 | 8 |
| 学習率 | 1e-6 |
| エポック数 | 1 |
| 反復回数 | 2 |
| プロンプト毎の生成数 | 8 |
| 最大生成長 | 1024トークン |
| 温度 | 1.0 |
| Beta (KLペナルティ) | 0.02 |
| WandBプロジェクト | qwenomni-grpo |


---

## 5. 報酬関数

### 5.1 Incremental DST 報酬（`DSTRewardIncremental`）

4つの要素の重み付き合計で報酬を計算する。

| 要素 | 重み | 説明 |
|------|------|------|
| Transcript WER Reward | 0.3 | `max(0, 1 - WER)`, 単語レベル編集距離ベース |
| Diff F1 Score | 0.5 | 差分操作の集合レベルF1スコア |
| Exact Match Bonus | 0.1 | 差分操作が完全一致した場合1.0 |
| Format Score | 0.1 | タグの存在+操作形式の正当性 |

**報酬範囲**: [0.0, 1.0]

**Format Scoreの内訳**:
- `<transcript>` タグの存在: +0.3
- `<answer>` タグの存在: +0.3
- 差分操作の形式が正しい（空も許容）: +0.4

**報酬計算例**:

| ケース | Transcript | Diff F1 | Exact Match | Format | 総報酬 |
|--------|-----------|---------|-------------|--------|--------|
| 完璧な予測 | 1.0 | 1.0 | 1.0 | 1.0 | **1.0** |
| ASR正解・DST部分一致 | 1.0 | 0.67 | 0.0 | 1.0 | 0.74 |
| ASR半分・DST正解 | 0.5 | 1.0 | 1.0 | 1.0 | 0.85 |
| フォーマットのみ正解 | 0.0 | 0.0 | 0.0 | 1.0 | 0.10 |
| 完全に失敗 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |

### 5.2 Full-State DST 報酬（`DSTRewardFullState`）

| 要素 | 重み | 説明 |
|------|------|------|
| Transcript WER Reward | 0.3 | 同上 |
| Slot F1 Score | 0.4 | スロットレベルF1（予測状態 vs 正解状態） |
| JGA (Joint Goal Accuracy) | 0.2 | 状態の完全一致 |
| Format Score | 0.1 | タグ + 有効なJSON形式 |

---

## 6. 推論と評価

### 6.1 推論モード

**Oracle モード**: 正解の対話履歴を使用（理想的条件での性能上限）

**Predicted モード**: モデルの予測を次ターンに引き継ぐカスケード評価（実用的な性能）
1. 対話IDごとにサンプルをグループ化
2. ターン順に逐次処理
3. 予測された転写をテキスト履歴に追加
4. Incremental: 予測された差分操作を適用して状態を更新 → 次ターンの入力に使用
5. Full-State: 転写のみ伝播（状態は毎ターン独立に生成）

### 6.2 推論エンジン

- **vLLM** (v0.15.1) を使用
- bfloat16精度、LoRAアダプタ対応
- GreedyDecoding（温度 0.0）
- 最大生成長: 1024トークン
- GPU メモリ使用率: 0.9

### 6.3 評価指標

| 指標 | 説明 |
|------|------|
| **Transcript WER** | 単語レベル編集距離 / 正解単語数（低いほど良い） |
| **JGA (Joint Goal Accuracy)** | 対話状態が完全に一致したターンの割合 |
| **Slot F1** | スロットレベルのF1スコア |

### 6.4 実験結果

#### Incremental DST — SFT + GRPO (Predicted モード)

| 指標 | SFT (checkpoint-4800) | GRPO (checkpoint-4800) |
|------|----------------------|----------------------|
| Transcript WER | ※oracle/predicted推論のみ実施 | 0.2168 |
| JGA | — | 0.4858 |
| Slot F1 | — | 0.8587 |
| 評価ターン数 | — | 17,782 |

※ SFTモデルでは推論ログにメトリクスの出力なし（oracle推論のみ実施）。GRPOモデルの結果はpredictedモードでのカスケード評価。

---

## 7. プロジェクト構成

```
qwenomni/
├── data/                                    # 学習・評価データ
│   ├── dapo_train.jsonl                     # GRPO学習データ
│   ├── dapo_val.jsonl                       # GRPO検証データ
│   ├── incremental_baseline_sft_test.jsonl  # テストデータ (Incremental)
│   ├── sft_train.jsonl                      # SFT学習データ（変換後）
│   └── sft_val.jsonl                        # SFT検証データ（変換後）
│
├── src/swift_plugin/
│   └── dapo_reward.py                # カスタム報酬関数 (DSTRewardIncremental, DSTRewardFullState)
│
├── system_prompt/
│   ├── incremental_baseline_sft.txt  # Incremental DST用システムプロンプト
│   └── fullstate_baseline.txt        # Full-State DST用システムプロンプト
│
├── scripts/
│   ├── train_sft.sh                  # SFT学習 (Incremental)
│   ├── train_grpo.sh                 # GRPO学習 (Incremental)
│   ├── train_sft_fullstate.sh        # SFT学習 (Full-State)
│   ├── train_grpo_fullstate.sh       # GRPO学習 (Full-State)
│   ├── infer_vllm.py                 # vLLM推論 (Incremental)
│   ├── infer_fullstate_vllm.py       # vLLM推論 (Full-State)
│   ├── infer_predicted.sh            # 推論実行スクリプト (Predicted)
│   ├── infer_fullstate_oracle.sh     # 推論実行スクリプト (Full-State Oracle)
│   ├── infer_fullstate_predicted.sh  # 推論実行スクリプト (Full-State Predicted)
│   ├── eval.py                       # 評価スクリプト (Incremental)
│   ├── eval_fullstate.py             # 評価スクリプト (Full-State)
│   ├── convert_to_sft.py             # GRPO→SFT形式変換
│   ├── prepare_fullstate_data.py     # Incremental→Full-State形式変換
│   └── sample_val.py                 # 検証データサンプリング
│
├── tests/
│   └── test_reward.py                # 報酬関数の単体テスト（30テスト）
│
├── logs/                             # 学習・推論ログ
│   ├── sft_20260212_172342.log
│   ├── grpo_20260213_162712.log
│   ├── sft_fullstate_20260215_*.log
│   └── infer_predicted_20260215_*.log
│
├── output/                           # チェックポイント・推論結果
│   ├── sft_incremental_dst/
│   ├── grpo_incremental_dst/
│   ├── sft_fullstate_dst/
│   └── vllm_inference_results/
│
├── docs/
│   ├── incremental_dst.md            # Incremental DST 技術仕様
│   └── research_overview.md          # 本ドキュメント
│
└── pyproject.toml                    # 依存関係定義
```

---

## 8. データパイプライン

### 8.1 データ形式の変換フロー

```
dapo_train.jsonl (GRPO形式)
    │
    ├──→ convert_to_sft.py ──→ sft_train.jsonl (SFT形式)
    │         └── assistant メッセージとして solution を追加
    │
    └──→ prepare_fullstate_data.py ──→ fullstate_dapo_train.jsonl (Full-State GRPO形式)
              │    └── [Previous State] 除去、システムプロンプト置換、
              │        solution を全状態JSONに変更
              │
              └──→ convert_to_sft.py ──→ fullstate_sft_train.jsonl (Full-State SFT形式)
```

### 8.2 GRPO形式のデータ構造

```json
{
  "messages": [
    {"role": "system", "content": "<システムプロンプト>"},
    {"role": "user", "content": "[Dialogue History]\n...\n[Previous State]\n...\n[New Audio]\n<audio>"}
  ],
  "audios": ["path/to/audio.wav"],
  "solution": "<transcript>...</transcript>\n<answer>...</answer>",
  "belief_state": "{\"domain\":{\"slot\":\"value\"}}",
  "prev_belief_state": "{}"
}
```

### 8.3 SFT形式のデータ構造

```json
{
  "messages": [
    {"role": "system", "content": "<システムプロンプト>"},
    {"role": "user", "content": "[Dialogue History]\n...\n[Previous State]\n...\n[New Audio]\n<audio>"},
    {"role": "assistant", "content": "<transcript>...</transcript>\n<answer>...</answer>"}
  ],
  "audios": ["path/to/audio.wav"]
}
```

---

## 9. 学習フロー図

```
┌─────────────────────────────────────────────────────────────────────┐
│                      GRPO 学習ループ                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌────────────────────┐    ┌───────────────┐    │
│  │ 学習データ   │───→│ Qwen2.5-Omni-7B   │───→│ 生成サンプル   │    │
│  │ (prompt)    │    │  + LoRA Adapter    │    │ (8個/prompt)  │    │
│  └─────────────┘    └────────────────────┘    └───────┬───────┘    │
│                                                       │            │
│                                                       ↓            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              DSTRewardIncremental / FullState                │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │  1. <transcript> から転写テキストを抽出                       │   │
│  │  2. <answer> から差分操作/JSON状態を抽出                      │   │
│  │  3. 各報酬要素を計算:                                        │   │
│  │     - Transcript WER Reward (0.3)                           │   │
│  │     - Diff F1 / Slot F1 (0.5 / 0.4)                        │   │
│  │     - Exact Match / JGA (0.1 / 0.2)                         │   │
│  │     - Format Score (0.1)                                     │   │
│  │  4. 総合報酬 [0, 1] を返却                                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      GRPO 最適化                             │   │
│  │  - 8サンプルから相対的な優劣でポリシーを更新                    │   │
│  │  - Clip-Higher, Dynamic Sampling                            │   │
│  │  - KLペナルティ (β=0.02) でSFTからの逸脱を抑制               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 10. 主要な依存関係

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| ms-swift | ≥ 3.12.4 | 学習フレームワーク（SFT/GRPO） |
| vllm | ≥ 0.13.0 | 推論エンジン |
| torch | ≥ 2.9.0 | 深層学習フレームワーク |
| torchaudio | ≥ 2.9.0 | 音声処理 |
| peft | ≥ 0.11.0 | LoRAアダプタライブラリ |
| deepspeed | ≥ 0.18.3 | 分散学習 |
| flash-attn | 2.8.3 | 高速Attention計算 |
| bitsandbytes | ≥ 0.49.0 | 4-bit量子化 |
| qwen-omni-utils | — | Qwen Omniモデルユーティリティ |
| wandb | ≥ 0.23.1 | 実験トラッキング |

---

## 11. 再現手順

### 11.1 Incremental DST

```bash
# 1. SFT学習
bash scripts/train_sft.sh

# 2. GRPO学習（SFTチェックポイントから開始）
bash scripts/train_grpo.sh

# 3. 推論（Predictedモード）
bash scripts/infer_predicted.sh
```

### 11.2 Full-State DST

```bash
# 1. SFT学習
bash scripts/train_sft_fullstate.sh

# 2. GRPO学習
bash scripts/train_grpo_fullstate.sh

# 3. 推論（Oracle / Predicted）
bash scripts/infer_fullstate_oracle.sh
bash scripts/infer_fullstate_predicted.sh
```

---

## 12. 実験タイムライン

| 日付 | 実験 | 備考 |
|------|------|------|
| 2026-02-12 | SFT (Incremental) | v6, 7,470ステップ |
| 2026-02-13 | GRPO (Incremental) | v8, checkpoint-4800から評価 |
| 2026-02-14 | 推論 (Oracle / Predicted) | SFTチェックポイントで推論 |
| 2026-02-15 | 推論 (GRPO Predicted) | WER=0.2168, JGA=0.4858, F1=0.8587 |
| 2026-02-15 | SFT (Full-State) | v1, 学習開始 |
| 2026-02-15 | GRPO (再実行) | 追加の学習実行 |
