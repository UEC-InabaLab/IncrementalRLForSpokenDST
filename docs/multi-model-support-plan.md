# 複数モデル対応 実装方針

目的: 現在Qwen2.5-Omni-7Bで行っているincremental Spoken DST（SFT→GRPOの
2段階学習）を、以下の追加モデルでも評価・可能であれば学習できるようにする。

1. **MiniCPM-o**（2.6 / 4.5）— OpenBMB
2. **Audio Flamingo 3** — NVIDIA
3. **Kimi-Audio** — MoonshotAI

このドキュメントは各モデルの学習・推論まわりの調査結果、モデルごとの統合方針、
段階的な実装順序をまとめたものです。固定仕様ではなく、実装を進める中で判明した
事実に応じて随時更新してください。

## なぜ3モデルで方針が異なるのか

このプロジェクトの学習パイプラインは全面的に
[ms-swift](https://github.com/modelscope/ms-swift)のCLI（`swift sft` /
`swift rlhf --rlhf_type grpo`）に依存しており、ms-swiftが明示的に登録している
モデルでしか動きません。ms-swift自体の対応モデル一覧
（`docs/source_en/Instruction/Supported-models-and-datasets.md`）を直接確認した結果:

| モデル | ms-swiftのレジストリに有無 | HF `transformers`のAutoModelクラス | 学習方法 |
|---|---|---|---|
| MiniCPM-o-2_6 / 4_5 | **あり**（`model_type: minicpmo`） | あり（ms-swift独自ローダ経由） | ms-swiftのSFT/GRPO CLI、Qwen2.5-Omni-7Bと同じ手順 |
| Audio Flamingo 3 | なし | **あり** — `AudioFlamingo3ForConditionalGeneration`、標準の`PreTrainedModel`、HF公式ドキュメントに学習例あり | 独自SFTスクリプト（HF `Trainer` + PEFT LoRA）。GRPOは独自ロールアウトループ、またはTRLの`GRPOTrainer`（このアーキテクチャでの動作は未検証） |
| Kimi-Audio | なし | なし — 独自の`modeling_kimia.py`、AutoModel未登録 | MoonshotAI公式のフルパラメータDeepSpeedファインチューニングスクリプト（`finetune_codes/`）あり。LoRA対応は未確認。GRPOは完全に独自のロールアウトループが必要 |

この違いにより、既存の`train_sft.sh` / `train_grpo.sh` / `infer.py`にそのまま
組み込めるのはMiniCPM-oのみです。Audio Flamingo 3とKimi-Audioは、ms-swiftを
経由しない独自の学習・推論コードパスが必要になります。

## 全モデルで共通利用できる部分

パイプラインの以下の部分はアーキテクチャに依存せず、**変更不要**です。

- **データ準備**（`scripts/train/prepare_data*.py`, `dst_common.py`）:
  出力は素のGRPO形式JSONL（`messages` / `audios` / `solution` /
  belief_state系フィールド）。どのモデルの学習スクリプトもこのJSONLを読み込み、
  そのモデルのプロセッサが期待するチャット形式に変換すればよい。
- **報酬計算**（`src/reward.py`）: `extract_transcript`, `extract_answer`,
  `compute_diff_f1`, `compute_transcript_wer_reward`などは、デコード済みの
  テキスト文字列だけを扱う。ms-swiftのGRPO CLIは既に`--external_plugins`
  経由でこれらを利用しており、Audio Flamingo 3 / Kimi-Audio用の独自GRPOループも
  ロジックを再実装せず、これらの関数をPythonの報酬コールバックとして直接
  importして使うべき。
- **評価**（`scripts/eval/eval.py`）: `prediction` / `solution` /
  belief_state系フィールドを持つpredictions JSONLを読み込む、モデル非依存の
  仕組み。各モデルの推論スクリプトがこの形式で出力しさえすればよい。

## モデルごとの統合方針

### 1. MiniCPM-o — 既存のms-swiftパイプラインをそのまま流用

最もリスクが低いため、最初に着手し、3モデル比較の土台を早期に作る。

- `train_sft.sh` / `train_grpo.sh` / `infer_oracle.sh` / `infer_predicted.sh`
  の既存`MODEL_PATH`環境変数に`OpenBMB/MiniCPM-o-2_6`（または`-4_5`）を
  指定するだけで動く可能性が高く、スクリプト自体の変更は不要かもしれない。
- **先に検証すべき点**: MiniCPM-oのチャットテンプレートが、このプロジェクトの
  GRPO形式（`<audio>`プレースホルダー付きのcontent文字列）をQwen2.5-Omniと
  同じように受け付けるか、それとも別のマルチモーダルcontent構造が必要か。
  本格的な学習前に、1サンプルでのスモークテストで確認する。
- 依存関係の注意点: `minicpmo4_5`はms-swiftのレジストリ上`transformers==4.51.3`
  への厳密な固定が必要。`qwen2_5_omni`が要求するバージョン（`transformers>=4.50`）
  と競合しないか確認し、競合する場合はMiniCPM-o用に別のuv環境が必要になる。

### 2. Audio Flamingo 3 — `transformers`をベースにした独自学習スクリプト

- **推論**: vLLMがネイティブ対応済み
  （`LLM(model="nvidia/audio-flamingo-3-hf")`）なので、
  `scripts/infer/infer.py`はモデルごとの分岐（または`scripts/infer/infer_af3.py`
  のような別スクリプト）としてメッセージ構築・カスケードロジックの大部分を
  流用できる見込み。モデル固有部分は音声content partsの構築方法のみ。
- **SFT**: ms-swift経由の方法が無いため、HF `Trainer` + PEFT LoRAを使った
  独自スクリプトを書く。コミュニティの参考実装
  [`Deep-unlearning/Finetune-AudioFlamingo3`](https://github.com/Deep-unlearning/Finetune-AudioFlamingo3)
  （言語モデルのattention/FFN層にLoRA、音声エンコーダは凍結）に倣う。
  HF公式ドキュメントに記載されているloss計算の契約:
  ```python
  inputs = processor.apply_chat_template(conversation, tokenize=True,
      add_generation_prompt=True, return_dict=True, output_labels=True)
  loss = model(**inputs).loss
  loss.backward()
  ```
- **GRPO**: 優先順に2案を検証する。
  1. TRLの`GRPOTrainer` — このアーキテクチャでの動作は未検証。TRLのVLM対応は
     「全てのVLMで動作保証はない」との記載があるため、まず小規模なスパイク
     （少量ロールアウトを実行し、出力が妥当か・報酬シグナルが機能するか確認）
     をしてから本格投資するかを判断する。
  2. TRLで動かない場合は、独自のロールアウトループ（補完文を生成→
     `src/reward.py`でスコアリング→方策勾配loss計算を手動実装）。コード量は
     増えるが制御性は高い。
- AF3のプロセッサには10分／20ウィンドウの音声長上限があるが、本プロジェクトの
  各サンプルは短いユーザーターン単発なので基本的に無関係。長尺音声にこの
  パイプラインを転用する場合のみ考慮すればよい。

### 3. Kimi-Audio — 最も工数が大きい。研究的なスパイクとして扱う

- **アーキテクチャ上の注意点**: Kimi-Audioはテキストと離散音声セマンティック
  トークンを**並行ヘッドで**生成するアーキテクチャであり、単一のテキストのみの
  causal LMヘッドではない。本格的な学習投資の前に、テキストのみの出力を
  強制／抽出する方法（transcript+diff操作のみ必要で、音声生成能力は不要）を
  確認する必要がある。学習時に音声トークンヘッドのlossをマスク／無視する必要が
  あるか、あるいは`kimia_infer`側に既にテキストのみのデコードモードがあるかを
  確認する。
- **SFT**: MoonshotAI公式が`finetune_codes/`にフルパラメータ・DeepSpeed
  ZeRO-2/3ベースのファインチューニング手順を提供している（LoRA対応は未確認）。
  独自のJSONLスキーマを使用:
  ```json
  {"task_type": "understanding", "conversation": [
    {"role": "user", "message_type": "text", "content": "..."},
    {"role": "user", "message_type": "audio", "content": "path/to.wav"},
    {"role": "assistant", "message_type": "text", "content": "..."}
  ]}
  ```
  このプロジェクトのGRPO形式JSONLからこのスキーマへの変換スクリプトが必要
  （`convert_to_sft.py`と同様の位置づけ）。学習起動スクリプトも彼らの
  `finetune_ds.sh`をベースに作成する。
- 既知の問題点: [MoonshotAI/Kimi-Audio#109](https://github.com/MoonshotAI/Kimi-Audio/issues/109)
  にて、ファインチューニング初期化時に`transformers==4.52.4`との非互換
  （`ALL_PARALLEL_STYLES`関連の`TypeError`）が報告されている。着手前に
  古い／動作確認済みの`transformers`バージョンに固定すること。リポジトリの
  `requirements.txt`や新しいissueコメントで解決済みバージョンを確認する。
- **推論**: 候補は2つ — MoonshotAI公式の`kimia_infer`ライブラリ（公式が推奨する
  経路）か、vLLMの`kimi_audio`モデル実行コード（vLLMのコードベースには存在
  するが、Moonshot公式ドキュメントが推奨する経路ではないため実験的として
  扱う）。`infer.py`の既存構造をそのまま活かせるため、まずvLLMを試し、
  出力がおかしければ`kimia_infer`にフォールバックする。
- **GRPO**: 独自アーキテクチャ・並行出力ヘッドのため、既存のRLトレーナーは
  そのままでは動かない。完全に独自のロールアウト＋報酬＋方策勾配ループが
  必要。ストレッチゴールとして扱い、Audio Flamingo 3 / MiniCPM-oの進捗を
  ブロックしないこと。

## リポジトリ構成の方針

モデルが増えるにつれ`scripts/train/`・`scripts/infer/`直下がサフィックス付き
ファイルだらけで読みにくくなるのを避けるため、ms-swift経由でない各モデルの
スクリプトはサブディレクトリにまとめる:

```
scripts/
  train/
    dst_common.py                 # 共通、変更なし
    prepare_data*.py               # 共通、変更なし
    train_sft.sh / train_grpo.sh   # ms-swift経路: MODEL_PATH=... でMiniCPM-oも選択
    audio_flamingo3/
      train_sft.py                 # HF Trainer + PEFT LoRA
      train_grpo.py                # TRLまたは独自ロールアウトループ
    kimi_audio/
      convert_to_kimia_format.py   # GRPO JSONL -> Kimi-Audioファインチューニング形式
      train_sft.sh                 # 彼らのfinetune_ds.shをラップ
  infer/
    infer.py                       # ms-swift/vLLM経路: Qwen2.5-Omni, MiniCPM-o
    audio_flamingo3/infer.py
    kimi_audio/infer.py
```

どちらの新モデル用サブディレクトリも、`prepare_data_dstc11.py`が既に使っている
サイドバイサイドimportのパターンで`dst_common.py`をimportし、`src/reward.py`を
直接importする（ロジックを重複させない）。

## 依存関係・環境の分離

モデル間でバージョン要件が競合する可能性が高い:

- `qwen2_5_omni`（現行）: `transformers>=4.50`
- `minicpmo4_5`: `transformers==4.51.3`（厳密固定）
- Audio Flamingo 3: `AudioFlamingo3ForConditionalGeneration`を含む
  `transformers`リリースが必要（モデルドキュメント上v4.57+/v5.x系 — 正確な
  最小バージョンは要確認）
- Kimi-Audio: 独自の`modeling_kimia.py`が正しく初期化されるには
  `transformers`が4.52.4より*古い*バージョンである必要（上記issueより）

推奨: 4モデル全てを1つのuv環境に無理に収めない。Audio Flamingo 3と
Kimi-Audioにはそれぞれ専用のロックファイル／venv
（例: `pyproject.audio-flamingo3.toml`、Kimi-Audio用に彼らのリポジトリに
準拠した`requirements.txt`）を用意し、既存のルート`pyproject.toml`は
Qwen2.5-Omni-7B + MiniCPM-o用として維持する（この2つのtransformersピンが
競合しないかは先に確認する）。

## 段階的な実装順序

1. **スパイク（3モデル共通）**: モデルごとに隔離したvenvで、事前学習済み
   チェックポイントをロードし、実際のSpokenWOZ/DSTC-11サンプル1件
   （テキスト履歴＋音声1ファイル）で1回のforward passを実行し、出力が
   妥当か確認する。低コストで、チャットテンプレート／データ形式の互換性を
   本格投資の前に検証できる。
2. **MiniCPM-o**: 既存の`train_sft.sh`/`train_grpo.sh`/`infer.py`に
   `MODEL_PATH`経由で組み込み、小規模SFTスモークテストを実行、
   Qwen2.5-Omni-7BベースラインとJGA/Slot F1/WERを比較する。
3. **Audio Flamingo 3**: 独自SFTスクリプト（PEFT経由のLoRA）を構築し、
   oracle-modeでのベースライン評価値を取得。その後GRPOをスパイク
   （まずTRL、ダメなら独自ループ）。
4. **Kimi-Audio**: テキストのみ出力の問題が解決し次第、公式
   `finetune_codes/`をSFT用に適応。推論経路（vLLM vs `kimia_infer`）を
   確定。GRPOは後回し／ストレッチゴール。

## 実装中に解決すべきリスク・未解決の疑問

- 各モデルのマルチモーダルチャットテンプレートが「テキスト履歴＋最新ターン
  のみ音声」という本プロジェクトのincremental DST入力形式に対応しているか、
  それとも音声が出現した全ターンでインラインの音声を期待するか。
  Audio Flamingo 3自身のマルチターン例からは、ターンごとにテキスト／音声の
  content partsを自由に組み合わせられそうだが、MiniCPM-oとKimi-Audioは未検証。
- Audio Flamingo 3（約8B）・Kimi-Audio（約10B）のGPUメモリ所要量が、現行の
  Qwen2.5-Omni-7B学習のGPU予算（8GPU、学習6＋vLLMロールアウト2）と比べて
  どうか — おそらく同程度だが未確認。
- Audio Flamingo 3（NVIDIA）・Kimi-Audio（コードはApache 2.0、重みは
  Qwen2.5-7B由来）のライセンス条件が、想定している論文発表／公開形態と
  両立するか、結果を報告する前に確認する。
- Kimi-Audioのテキスト／音声トークン並行生成: 学習・推論の両方で、この
  タスクに必要な`<transcript>`/`<answer>`テキストのみを確実に出力させ、
  音声トークンヘッドを抑制する方法を確認する。
