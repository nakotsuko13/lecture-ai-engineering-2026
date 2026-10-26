#!/usr/bin/env python
"""第2回 演習: LoRA で回答形式を学習し、学習前後を比較する（DGX Spark ローカル実行版）。

notebooks/lora_format_tuning.ipynb（版 1.1、Colab T4 向け）のセクション 0〜7 を
本マシン（DGX Spark / GB10）の venv で動くようにスクリプト化したもの。

ノートブックからの変更点（環境差の吸収のみで、学習・評価のロジックは同一）:
- セクション 0 の %pip 導入は削除（venv に事前導入済み、版は PINS どおり）
- USE_DRIVE / Google Drive マウントを削除（ローカル保存のみ）
- CONFIG の work_dir を /content/qlora_lab → lectures/02/work/exercise に変更
- display() を print(DataFrame.to_string()) に置換
- 進捗が分かるよう flush 付きの log を追加
"""

import json
import re
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch
import transformers
import peft
import trl
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

pd.set_option("display.max_colwidth", None)

HERE = Path(__file__).resolve().parent
WORK = HERE / "exercise"
WORK.mkdir(parents=True, exist_ok=True)


def log(*args):
    print(*args, flush=True)


def show(df: pd.DataFrame):
    log(df.to_string(index=False))


# ===== セクション 0: 環境の確認 =====
try:
    gpu_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
        capture_output=True, text=True,
    ).stdout
except FileNotFoundError:
    gpu_info = ""
log("GPU:", gpu_info.strip() or "GPU が見つかりません")
assert torch.cuda.is_available(), "CUDA が使えません。GPU 環境を確認してください"

# ===== セクション 1: 設定 =====
CONFIG = {
    "base_model": "Qwen/Qwen3.5-0.8B",   # 小さな指示チューニング済みモデル（Apache-2.0）
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",  # いつ実行しても同じ重み
    "seed": 42,
    "num_train_epochs": 20,  # 学習データ（14 件）を 20 周 → 合計 40 ステップ
    "warmup_steps": 6,
    "lr": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_seq_len": 512,
    "max_new_tokens": 96,
    "gen": {"do_sample": False},   # 評価は貪欲生成で固定
    "work_dir": str(WORK),
}
log("CONFIG:\n" + json.dumps(CONFIG, ensure_ascii=False, indent=2))

# ===== セクション 2: 架空 FAQ と教師データの生成 =====
FAQ = {
    "deadline": "課題の提出期限は 9 月 30 日 23:59（日本時間）です。",
    "destination": "提出先は学習ポータルの「課題」欄です。",
    "format": "提出形式は PDF 1 ファイルで、ファイル名は 学籍番号_課題番号.pdf です。",
    "office_hours": "質問は毎週火曜 17:00〜18:00 のオフィスアワーで受け付けます。",
    "late": "期限後の提出は、期限から 48 時間以内なら減点のうえ受け付けます。",
}
FAQ_TEXT = "\n".join(f"- {v}" for v in FAQ.values())

# 根拠あり: (FAQ キー, 質問テンプレート群, 正解の answer に含まれるべき語)
GROUNDED = {
    "deadline": (["課題の提出期限はいつですか", "いつまでに出せばいいですか", "締切を教えてください", "提出の期限は？"], ["9 月 30 日", "23:59"]),
    "destination": (["提出先はどこですか", "どこにアップロードしますか", "課題はどこから出しますか", "提出場所は？"], ["学習ポータル", "課題"]),
    "format": (["提出形式を教えてください", "ファイル名の決まりはありますか", "何のファイル形式で出しますか", "PDF 以外でも出せますか"], ["PDF"]),
    "office_hours": (["質問はいつできますか", "オフィスアワーはいつですか", "先生に質問する時間はありますか", "相談できる時間帯は？"], ["火曜", "17:00"]),
    "late": (["遅れて出したらどうなりますか", "期限を過ぎた場合は受け付けますか", "遅延提出のルールは？", "締切後の扱いを教えてください"], ["48 時間", "減点"]),
}
# 根拠なし: FAQ に無いので needs_review=true が正解
UNGROUNDED = [
    "再提出はできますか", "欠席した回の扱いを教えてください", "他の受講者の点数は見られますか",
    "成績はいつ出ますか", "提出期限は延長できますか", "グループでの提出は可能ですか",
    "課題の配点はどうなっていますか", "講義の録画は公開されますか",
]

# test 専用の追加質問（train / valid には入れない）
TEST_EXTRA_GROUNDED = {
    "deadline": ["提出期限の日付と時刻を教えてください", "課題の締切は何月何日の何時ですか"],
    "destination": ["課題の提出はどこで行いますか", "提出物を出す先を教えてください"],
    "format": ["提出するファイルの形式は何ですか", "提出ファイルは何形式で、いくつ出しますか"],
    "office_hours": ["オフィスアワーの曜日と時間を教えてください", "質問を受け付けてもらえるのは何曜日の何時からですか"],
    "late": ["提出が締切に間に合わなかったらどうなりますか", "期限後に提出した課題は受理されますか"],
}
TEST_EXTRA_UNGROUNDED = [
    "課題の文字数に制限はありますか", "提出した課題へのフィードバックはもらえますか",
    "参考文献の書き方に指定はありますか", "オフィスアワーは事前予約が必要ですか",
]
UNGROUNDED_ANSWER = "この FAQ には記載がありません。運営に確認が必要です。"


def make_example(question: str, answer: str, needs_review: bool) -> dict:
    target = json.dumps({"answer": answer, "needs_review": needs_review}, ensure_ascii=False)
    return {"question": question, "faq": FAQ_TEXT, "target": target, "needs_review": needs_review}


def split_templates(templates):
    # テンプレートの先頭 2 つを train、3 つ目を valid、4 つ目を test に固定（言い換えを跨がせない）
    return templates[:2], templates[2:3], templates[3:4]


train, valid, test, gold = [], [], [], {}
for key, (templates, must_include) in GROUNDED.items():
    tr, va, te = split_templates(templates)
    for q in tr:
        train.append(make_example(q, FAQ[key], False))
    for q in va:
        valid.append(make_example(q, FAQ[key], False))
    for q in te + TEST_EXTRA_GROUNDED[key]:
        test.append(make_example(q, FAQ[key], False))
        gold[q] = {"needs_review": False, "must_include": must_include}
for i, q in enumerate(UNGROUNDED):
    ex = make_example(q, UNGROUNDED_ANSWER, True)
    bucket = [train, train, valid, test][i % 4]   # 根拠なしの質問は 4 問ごとに train 2 / valid 1 / test 1
    bucket.append(ex)
    if bucket is test:
        gold[q] = {"needs_review": True, "must_include": ["確認"]}
for q in TEST_EXTRA_UNGROUNDED:
    test.append(make_example(q, UNGROUNDED_ANSWER, True))
    gold[q] = {"needs_review": True, "must_include": ["確認"]}

# train は 14 件（根拠あり 10、根拠なし 4）。同じ例を何回学習するかは num_train_epochs で決める
all_questions = [r["question"] for r in train + valid + test]
assert len(set(all_questions)) == len(all_questions), "質問が train / valid / test のどこかで重複しています"
log(f"train={len(train)} valid={len(valid)} test={len(test)} gold={len(gold)}")
log("学習例の 1 件 | 質問:", train[0]["question"], "| 望ましい出力:", train[0]["target"])
(WORK / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train), encoding="utf-8")
(WORK / "test.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in test), encoding="utf-8")
(WORK / "gold.json").write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")

# ===== セクション 3: モデルの読み込み =====
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
# 量子化なし。重みは 32bit のまま保持する
model = AutoModelForCausalLM.from_pretrained(
    CONFIG["base_model"], revision=CONFIG["revision"], dtype=torch.float32
).to(DEVICE)
model.config.use_cache = True
BASE_MEMORY_GB = round(model.get_memory_footprint() / 1e9, 2)
log("loaded:", CONFIG["base_model"], "| device:", DEVICE, "| 重みのメモリ(GB):", BASE_MEMORY_GB)

# 形式はプロンプトで指示している。モデルがこれを守れるかが学習前の確認点
SYSTEM = "あなたは講座 FAQ の回答係です。与えられた FAQ だけを根拠に、必ず JSON 1 行で答えてください。形式: {\"answer\": 文字列, \"needs_review\": 真偽値}。FAQ に記載が無いことは answer に『確認が必要』と書き、needs_review を true にします。"


def build_messages(example):
    user = f"FAQ:\n{example['faq']}\n\n質問: {example['question']}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def generate(model_obj, example):
    # enable_thinking=False: 思考ブロックを空にして回答だけを生成させる
    prompt = tok.apply_chat_template(build_messages(example), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt").to(model_obj.device)
    with torch.no_grad():
        out = model_obj.generate(**inputs, max_new_tokens=CONFIG["max_new_tokens"], pad_token_id=tok.pad_token_id, **CONFIG["gen"])
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ===== 採点関数 =====
def check_format(text: str):
    """JSON として読め、必須キーと型が合うか。失敗理由も返す"""
    try:
        obj = json.loads(text)
    except Exception as e:
        return False, f"json_error: {type(e).__name__}", None
    if not isinstance(obj, dict):
        return False, "not_object", None
    if set(obj) - {"answer", "needs_review"}:
        return False, "extra_keys", obj
    if "answer" not in obj or "needs_review" not in obj:
        return False, "missing_key", obj
    if not isinstance(obj["answer"], str) or not isinstance(obj["needs_review"], bool):
        return False, "wrong_type", obj
    return True, "", obj


def check_content(obj, gold_entry):
    """内容の合否: needs_review が正解と一致し、必須語句を answer に含む"""
    if obj is None:
        return False
    if obj.get("needs_review") != gold_entry["needs_review"]:
        return False
    return all(w in obj.get("answer", "") for w in gold_entry["must_include"])


def mark(ok: bool) -> str:
    return "○" if ok else "×"


def evaluate(model_obj, label: str):
    """test の 21 問を生成して採点し、要約と質問ごとの結果を返す。原文は eval_<label>.json に保存する"""
    rows = []
    t0 = time.time()
    for ex in test:
        raw = generate(model_obj, ex)
        ok_fmt, reason, obj = check_format(raw)
        # 形式が壊れていても JSON らしい部分が読めるなら、内容だけは別に採点する
        loose = obj
        if not ok_fmt:
            m = re.search(r"\{.*\}", raw, re.S)
            try:
                loose = json.loads(m.group(0)) if m else None
            except Exception:
                loose = None
        ok_content = check_content(loose if isinstance(loose, dict) else None, gold[ex["question"]])
        rows.append({"question": ex["question"], "raw": raw, "format_ok": ok_fmt, "format_reason": reason,
                     "content_ok": ok_content, "both_ok": ok_fmt and ok_content,
                     "gold_needs_review": gold[ex["question"]]["needs_review"]})
        log(f"  [{label}] {len(rows)}/{len(test)}: {ex['question']} -> {mark(ok_fmt)}/{mark(ok_content)}")
    n = len(rows)
    summary = {"label": label, "n": n,
               "format_rate": sum(r["format_ok"] for r in rows) / n,
               "content_rate": sum(r["content_ok"] for r in rows) / n,
               "both_rate": sum(r["both_ok"] for r in rows) / n,
               "seconds": round(time.time() - t0, 1)}
    (WORK / f"eval_{label}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(summary, ensure_ascii=False))
    return summary, rows


# ===== 学習前評価 =====
log("===== 学習前評価 (before) =====")
before_summary, before_rows = evaluate(model, "before")
show(pd.DataFrame([{"質問": r["question"], "正解の needs_review": r["gold_needs_review"], "学習前の出力": r["raw"],
                    "形式": mark(r["format_ok"]), "形式の失敗理由": r["format_reason"], "内容": mark(r["content_ok"])}
                   for r in before_rows]))

# ===== セクション 4: LoRA 学習 =====
log("===== LoRA 学習 =====")


def to_chat(example):
    # 学習データは「入力（system + user）」と「望ましい出力（assistant）」の対。評価と同じ build_messages を使う
    return {"messages": build_messages(example) + [{"role": "assistant", "content": example["target"]}]}


train_ds = Dataset.from_list([to_chat(e) for e in train])
valid_ds = Dataset.from_list([to_chat(e) for e in valid])

model.config.use_cache = False
# Qwen3.5 は 24 層のうち 18 層が線形注意で、q/k/v/o_proj を持つのは残りの 6 層。MLP の gate/up/down_proj は全層にある
lora = LoraConfig(r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
USE_AMP = False  # 通常は変えない。True にすると計算だけ 16bit の混合精度になる
sft = SFTConfig(output_dir=str(WORK / "ckpt"), num_train_epochs=CONFIG["num_train_epochs"], per_device_train_batch_size=4,
                gradient_accumulation_steps=2, learning_rate=CONFIG["lr"], lr_scheduler_type="cosine",
                warmup_steps=CONFIG["warmup_steps"], logging_steps=10, save_steps=10, save_total_limit=2,
                fp16=USE_AMP, max_length=CONFIG["max_seq_len"], packing=False, report_to="none", seed=CONFIG["seed"])
trainer = SFTTrainer(model=model, args=sft, train_dataset=train_ds, eval_dataset=valid_ds, peft_config=lora, processing_class=tok)
trainer.model.print_trainable_parameters()
t0 = time.time()
trainer.train()
TRAIN_SECONDS = time.time() - t0
log(f"学習時間: {TRAIN_SECONDS:.0f} 秒 | ステップ数: {trainer.state.global_step} | エポック: {trainer.state.epoch:.1f}")
ADAPTER_DIR = WORK / "adapter"
trainer.model.save_pretrained(ADAPTER_DIR)
tuned = trainer.model
tuned.config.use_cache = True
_ = tuned.eval()  # 学習モードのままだと dropout が効き、評価の出力が揺れる
TRAIN_LOSS_LAST = trainer.state.log_history[-1].get("train_loss") if trainer.state.log_history else None
log("最終 train_loss:", TRAIN_LOSS_LAST)

# ===== セクション 5: 学習後評価 =====
log("===== 学習後評価 (after) =====")
after_summary, after_rows = evaluate(tuned, "after")
show(pd.DataFrame([before_summary, after_summary]).set_index("label")[["n", "format_rate", "content_rate", "both_rate", "seconds"]])
# 学習前は形式も内容も通ったのに、学習後に落ちた質問（退行）
regressed = [b["question"] for b, a in zip(before_rows, after_rows) if b["both_ok"] and not a["both_ok"]]
log("退行:", regressed)

show(pd.DataFrame([{"質問": b["question"], "正解の needs_review": b["gold_needs_review"],
                    "学習前の出力": b["raw"], "学習前 形式/内容": f"{mark(b['format_ok'])}/{mark(b['content_ok'])}",
                    "学習後の出力": a["raw"], "学習後 形式/内容": f"{mark(a['format_ok'])}/{mark(a['content_ok'])}"}
                   for b, a in zip(before_rows, after_rows)]))

# ===== セクション 6: 知識を問う質問 =====
log("===== 知識を問う質問 =====")
KNOWLEDGE_QUESTIONS = [
    "この講座の最終課題の締切はいつですか",       # FAQ に無い（架空）
    "担当教員の名前は何ですか",                     # FAQ に無い
    "次回の講義のテーマは何ですか",                 # FAQ に無い
    "課題の提出期限はいつですか",                   # FAQ にある（対照）
]


def classify(raw):
    ok, _, obj = check_format(raw)
    if not ok or obj is None:
        return "format_error"
    return "needs_review" if obj["needs_review"] else "asserted"


knowledge_rows = []
for q in KNOWLEDGE_QUESTIONS:
    ex = make_example(q, "", True)
    # 学習前の出力は adapter を一時的に無効にして取る
    with tuned.disable_adapter():
        before_raw = generate(tuned, ex)
    after_raw = generate(tuned, ex)
    knowledge_rows.append({"質問": q, "FAQ にある": q in gold, "学習前の分類": classify(before_raw), "学習前の出力": before_raw,
                           "学習後の分類": classify(after_raw), "学習後の出力": after_raw})
show(pd.DataFrame(knowledge_rows))

# ===== セクション 7: 比較表と提出 =====
log("===== result.json 作成 =====")
result = {
    "config": {k: v for k, v in CONFIG.items() if k not in ("work_dir", "drive_dir")},
    "versions": {m.__name__: m.__version__ for m in [transformers, peft, trl, torch]},
    "method": "LoRA（量子化なし）",
    "device": DEVICE,
    "before": before_summary, "after": after_summary,
    "train_loss_last": TRAIN_LOSS_LAST,
    "regressed": regressed,
    "knowledge": knowledge_rows,
    "executed_training_myself": True,
    "runtime": "Google Colab ではなくローカルマシン（DGX Spark / GB10, CUDA 13.0）で実行",
}
(WORK / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
log(json.dumps({k: v for k, v in result.items() if k != "knowledge"}, ensure_ascii=False, indent=2))
log("DONE: results in", WORK)



