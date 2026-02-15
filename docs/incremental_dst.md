# Incremental DST（増分対話状態追跡）実装ドキュメント

## 概要

Incremental DSTは、対話を1ターンずつ逐次処理し、**前ターンの対話状態から差分操作のみで状態を更新**する手法です。従来の累積手法（毎ターン全状態を予測）とは異なり、変更があったスロットのみを予測します。

---

## 入出力データ構造

### 学習データ形式（ms-swift GRPO形式）

**ファイル**: `data/dapo_train.jsonl`

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an incremental Spoken Dialogue State Tracker (DST).\n\nYou receive:\n1. Text dialogue history (previous turns)\n2. Previous dialogue state (JSON)\n3. Audio of the latest 2 turns (system + user)\n\nYour task:\n1. Transcribe the audio in <transcript>\n2. Output slot changes in <answer>\n\nSlot change format (one per line):\n- set(domain.slot=value): New slot\n- update(domain.slot=value): Changed value\n- delete(domain.slot): Removed slot\n\nExample output:\n<transcript>\nSystem: do you have a price preference.\nUser: i'd like something cheap.\n</transcript>\n<answer>set(restaurant.pricerange=cheap)</answer>"
    },
    {
      "role": "user",
      "content": "[Dialogue History]\nUser: hello is this customer service center.\n\n[Previous State]\n{}\n\n[New Audio]\n<audio>"
    }
  ],
  "audios": ["data/processed_audio_incremental_16k/MUL0003_1_2.wav"],
  "solution": "<transcript>\nSystem: yes, this is customer service center. how may i help.\nUser: well, i'm looking for a place to dine. do you have any recommendation for me.\n</transcript>\n<answer></answer>",
  "belief_state": "{}",
  "prev_belief_state": "{}"
}
```

#### フィールド説明

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `messages` | Array | 対話メッセージ（system + user） |
| `messages[0].content` | String | システムプロンプト |
| `messages[1].content` | String | ユーザー入力（履歴 + 前状態 + 音声プレースホルダー） |
| `audios` | Array[String] | 音声ファイルパスのリスト |
| `solution` | String | 正解出力（トランスクリプト + 差分操作） |
| `belief_state` | String (JSON) | **現ターンの対話状態**（報酬計算用） |
| `prev_belief_state` | String (JSON) | **前ターンの対話状態**（差分計算用） |

### ユーザー入力の構造

```
[Dialogue History]
User: hello is this customer service center.
System: yes, how may i help.

[Previous State]
{"restaurant":{"area":"centre"}}

[New Audio]
<audio>
```

| セクション | 説明 |
|-----------|------|
| `[Dialogue History]` | 音声より前のテキスト履歴（ターン順） |
| `[Previous State]` | 前ターン終了時点の対話状態（JSON） |
| `[New Audio]` | 2ターン音声（システム発話 + ユーザー発話） |

### モデル出力の構造

<transcript>
System: do you have a price preference.
User: i'd like something cheap.
</transcript>
<answer>set(restaurant.pricerange=cheap)</answer>


| タグ | 説明 |
|-----|------|
| `<transcript>` | 2ターン音声のトランスクリプト |
| `<answer>` | 差分操作（改行区切りで複数可、空も許容） |

### 差分操作の形式

| 操作 | 形式 | 例 |
|------|------|-----|
| **set** | `set(domain.slot=value)` | `set(restaurant.pricerange=cheap)` |
| **update** | `update(domain.slot=value)` | `update(hotel.stars=5)` |
| **delete** | `delete(domain.slot)` | `delete(hotel.parking)` |


## 強化学習（DAPO）の報酬関数

### 報酬関数クラス

**ファイル**: `src/swift_plugin/dapo_reward.py` - `DSTRewardIncremental`

```python
class DSTRewardIncremental(ORM):
    """
    Incremental DST Reward Function for ms-swift.

    Evaluates both ASR (transcript) and DST (diff operations) accuracy.
    """
```

### 報酬構成要素

| 要素 | 重み（デフォルト） | 説明 |
|------|-------------------|------|
| **transcript** | 0.3 | トランスクリプトの単語レベルF1（ASR精度） |
| **diff_f1** | 0.5 | 差分操作のF1スコア（DST精度） |
| **exact_match** | 0.1 | 差分操作の完全一致ボーナス |
| **format** | 0.1 | フォーマット正当性 |

**報酬範囲**: [0.0, 1.0]

### 各要素の計算詳細

#### 1. Transcript Score（ASR報酬）

```python
def compute_transcript_similarity(pred_transcript: str, gold_transcript: str) -> float:
    """単語レベルのF1スコアを計算"""
    # 小文字化 → 単語分割 → Counter
    pred_words = pred_transcript.lower().split()
    gold_words = gold_transcript.lower().split()

    # 共通単語数
    intersection = sum((Counter(pred_words) & Counter(gold_words)).values())

    precision = intersection / len(pred_words)
    recall = intersection / len(gold_words)
    f1 = 2 * precision * recall / (precision + recall)
    return f1
```

#### 2. Diff F1 Score（DST報酬）

```python
def compute_diff_f1(pred_ops, gold_ops) -> Tuple[float, float, float]:
    """差分操作のPrecision, Recall, F1を計算"""
    pred_set = ops_to_set(pred_ops)  # {(op, domain, slot, value), ...}
    gold_set = ops_to_set(gold_ops)

    correct = len(pred_set & gold_set)  # 正解した操作数
    precision = correct / len(pred_set)
    recall = correct / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
```

**例**:
- 予測: `{set(restaurant.area=centre), set(restaurant.food=chinese)}`
- 正解: `{set(restaurant.area=centre)}`
- Precision: 1/2 = 0.5, Recall: 1/1 = 1.0, F1 = 0.67

#### 3. Exact Match（完全一致ボーナス）

```python
exact_match = 1.0 if pred_set == gold_set else 0.0
```

予測と正解の差分操作が完全に一致した場合のみ1.0を付与。

#### 4. Format Score（フォーマット評価）

```python
def compute_incremental_format_score(response: str) -> float:
    score = 0.0

    # <transcript>タグの存在: +0.3
    if '<transcript>' in response and '</transcript>' in response:
        score += 0.3

    # <answer>タグの存在: +0.3
    if '<answer>' in response and '</answer>' in response:
        score += 0.3

        # 差分操作の形式が正しい: +0.4
        # 空 or 全行が set(...)/update(...)/delete(...) 形式
        if valid_format:
            score += 0.4

    return score  # 最大1.0
```

### 総合報酬の計算

```python
total_reward = (
    weights['transcript'] * transcript_score +    # 0.3 * [0-1]
    weights['diff_f1'] * diff_f1 +                # 0.5 * [0-1]
    weights['exact_match'] * exact_match +        # 0.1 * [0 or 1]
    weights['format'] * format_score              # 0.1 * [0-1]
)
# Clamp to [0, 1]
reward = min(max(total_reward, 0.0), 1.0)
```

### 報酬例

| ケース | transcript | diff_f1 | exact_match | format | 総報酬 |
|--------|------------|---------|-------------|--------|--------|
| 完璧な予測 | 1.0 | 1.0 | 1.0 | 1.0 | **1.0** |
| ASR正解・DST完全一致 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| ASR正解・DST部分一致 | 1.0 | 0.67 | 0.0 | 1.0 | 0.74 |
| ASR半分・DST正解 | 0.5 | 1.0 | 1.0 | 1.0 | 0.85 |
| フォーマットのみ正解 | 0.0 | 0.0 | 0.0 | 1.0 | **0.1** |
| 完全に失敗 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |



## データフロー図


┌─────────────────────────────────────────────────────────────────┐
│                        DAPO学習ループ                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │ 学習データ   │───→│ Audio-Flamingo-3 │───→│ 生成サンプル   │  │
│  │ (prompt)    │    │  + LoRA Adapter  │    │ (completion)  │  │
│  └─────────────┘    └──────────────────┘    └───────┬───────┘  │
│                                                      │          │
│                                                      ↓          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  DSTRewardIncremental                    │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  1. parse transcript from <transcript>                   │   │
│  │  2. parse diff ops from <answer>                        │   │
│  │  3. compute gold diff from prev_state → curr_state      │   │
│  │  4. calculate reward components:                         │   │
│  │     - transcript F1 (0.3)                               │   │
│  │     - diff F1 (0.5)                                     │   │
│  │     - exact match (0.1)                                 │   │
│  │     - format (0.1)                                      │   │
│  │  5. return total reward [0, 1]                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    GRPO/DAPO最適化                       │   │
│  │  - 複数サンプルから最良/最悪を選択                        │   │
│  │  - Clip-Higher, Dynamic Sampling                        │   │
│  │  - Overlong Filtering                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

