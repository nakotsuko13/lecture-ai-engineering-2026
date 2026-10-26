#!/usr/bin/env python
"""第2回 宿題（発展）: 講座の用語集から SFT データを作り、FAQ の代わりに学習して学習前後を比べる。

problem.md の宿題「発展」レベル:
- scripts/make_sft_dataset.py で講座の用語集から Q&A データ（messages 形式の JSONL）を作る
- FAQ の代わりに学習させる（JSON 形式と needs_review の設計は演習のまま使う）
- 評価用の質問（用語集にある語・言い回しを変えた質問・用語集にない語）と正解を自分で用意し、学習前後を比べる

演習（run_exercise.py）との違い:
- 入力に用語集・FAQ を渡さない（知識を重みに焼き込む方式。第3回の「検索するエージェント」との対比用）
- 学習データは用語集 58 語 × 2 テンプレート（make_sft_dataset.py の出力を JSON 形式に包む）＋
  「用語集にない語」への確認応答 20 件（needs_review: true の教師）
- 評価は自作の 15 問（用語集にある語 5・言い回し変え 5・用語集にない語 5）
"""

import json
import random
import re
import subprocess
import sys
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
REPO = HERE.parents[2]          # lectures/02/work -> repo root
WORK = HERE / "homework"
WORK.mkdir(parents=True, exist_ok=True)


def log(*args):
    print(*args, flush=True)


def show(df: pd.DataFrame):
    log(df.to_string(index=False))


# ===== ステップ 1: scripts/make_sft_dataset.py で用語集から Q&A データ（JSONL）を作る =====
GLOSSARY_JSONL = WORK / "glossary_sft.jsonl"
proc = subprocess.run(
    [sys.executable, str(REPO / "scripts" / "make_sft_dataset.py")],
    capture_output=True, text=True, check=True,
)
GLOSSARY_JSONL.write_text(proc.stdout, encoding="utf-8")
glossary_records = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
log(f"glossary_sft.jsonl: {len(glossary_records)} レコード")

# ===== ステップ 2: 演習と同じ JSON 形式（system + user + assistant）に変換する =====
SYSTEM = ("あなたは講座用語集の回答係です。講座の用語集について聞かれたら、必ず JSON 1 行で答えてください。"
          "形式: {\"answer\": 文字列, \"needs_review\": 真偽値}。"
          "用語集に記載が無いことは answer に『確認が必要』と書き、needs_review を true にします。")
NOT_IN_GLOSSARY_ANSWER = "この用語集には記載がありません。運営に確認が必要です。"


def to_chat(question: str, target: str):
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": question},
                         {"role": "assistant", "content": target}]}


# 用語ごと（2 テンプレートのペア単位）に train / valid へ振り分ける
rng = random.Random(42)
terms = {}
for rec in glossary_records:
    q = rec["messages"][0]["content"]
    a = rec["messages"][1]["content"]
    term = q.replace("「", "").split("」とは")[0] if q.startswith("「") else q.split("について簡単に")[0]
    terms.setdefault(term, []).append((q, a))
all_terms = sorted(terms)
valid_terms = set(rng.sample(all_terms, 8))
train, valid = [], []
for term, pairs in terms.items():
    for q, a in pairs:
        target = json.dumps({"answer": a, "needs_review": False}, ensure_ascii=False)
        (valid if term in valid_terms else train).append(to_chat(q, target))

# 「用語集にない語」への確認応答を教師に加える（needs_review: true の学習データ）
UNSEEN_TERMS_TRAIN = [
    "検索拡張生成", "強化学習", "ベクトルデータベース", "クロスバリデーション", "畳み込みニューラルネットワーク",
    "モンテカルロ法", "特徴量エンジニアリング", "分散表現", "アンサンブル学習", "オントロジー",
]
for term in UNSEEN_TERMS_TRAIN:
    for q in [f"{term}とは何ですか？", f"{term}について簡単に説明してください。"]:
        target = json.dumps({"answer": NOT_IN_GLOSSARY_ANSWER, "needs_review": True}, ensure_ascii=False)
        train.append(to_chat(q, target))

log(f"train={len(train)} valid={len(valid)} | valid に当たった用語: {sorted(valid_terms)}")

# ===== ステップ 3: 評価用の質問（15 問）と正解を自作する =====
# A: 用語集にある語を学習テンプレートと同じ言い回しで（暗記の確認）
# B: 用語集にある語を言い回しを変えて（一般化の確認）
# C: 用語集にない語（確認応答に変わるか。演習セクション 6 相当）
EVAL_QUESTIONS = [
    ("A_用語集にある語", "トークンとは何ですか？", False, ["最小単位"]),
    ("A_用語集にある語", "LoRAとは何ですか？", False, ["低ランク"]),
    ("A_用語集にある語", "エポックとは何ですか？", False, ["1周"]),
    ("A_用語集にある語", "ハルシネーションとは何ですか？", False, ["もっともらしく"]),
    ("A_用語集にある語", "MCP（Model Context Protocol）とは何ですか？", False, ["プロトコル"]),
    ("B_言い回しを変えた質問", "コンテキストウィンドウの意味を教えてください", False, ["上限"]),
    ("B_言い回しを変えた質問", "temperatureとはどういう設定ですか", False, ["ランダムさ"]),
    ("B_言い回しを変えた質問", "ガードレールという言葉の意味を説明してください", False, ["制約"]),
    ("B_言い回しを変えた質問", "事前学習は何を学習する段階ですか", False, ["トークン"]),
    ("B_言い回しを変えた質問", "過学習について教えてください", False, ["未知"]),
    ("C_用語集にない語", "RAGとは何ですか？", True, ["確認"]),
    ("C_用語集にない語", "埋め込みとは何ですか？", True, ["確認"]),
    ("C_用語集にない語", "RLHFとは何ですか？", True, ["確認"]),
    ("C_用語集にない語", "プロンプトエンジニアリングとは何ですか？", True, ["確認"]),
    ("C_用語集にない語", "拡散モデルとは何ですか？", True, ["確認"]),
]

CONFIG = {
    "base_model": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "seed": 42,
    "num_train_epochs": 10,   # train 120 件 → 1 エポック 15 ステップ、計 150 ステップ
    "warmup_steps": 15,
    "lr": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_seq_len": 512,
    "max_new_tokens": 96,
    "gen": {"do_sample": False},
    "valid_terms": sorted(valid_terms),
    "unseen_train_terms": UNSEEN_TERMS_TRAIN,
    "work_dir": str(WORK),
}
log("CONFIG:\n" + json.dumps(CONFIG, ensure_ascii=False, indent=2))
(WORK / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train), encoding="utf-8")
(WORK / "valid.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in valid), encoding="utf-8")

# ===== モデル読み込み =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    CONFIG["base_model"], revision=CONFIG["revision"], dtype=torch.float32
).to(DEVICE)
model.config.use_cache = True
log("loaded:", CONFIG["base_model"], "| device:", DEVICE, "| 重みのメモリ(GB):", round(model.get_memory_footprint() / 1e9, 2))


def build_messages(question):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]


def generate(model_obj, question):
    prompt = tok.apply_chat_template(build_messages(question), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt").to(model_obj.device)
    with torch.no_grad():
        out = model_obj.generate(**inputs, max_new_tokens=CONFIG["max_new_tokens"], pad_token_id=tok.pad_token_id, **CONFIG["gen"])
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ===== 採点（演習と同じ基準） =====
def check_format(text: str):
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


def mark(ok):
    return "○" if ok else "×"


def evaluate(model_obj, label: str):
    rows = []
    t0 = time.time()
    for category, q, gold_nr, must in EVAL_QUESTIONS:
        raw = generate(model_obj, q)
        ok_fmt, reason, obj = check_format(raw)
        loose = obj
        if not ok_fmt:
            m = re.search(r"\{.*\}", raw, re.S)
            try:
                loose = json.loads(m.group(0)) if m else None
            except Exception:
                loose = None
        if not isinstance(loose, dict):
            ok_content = False
        else:
            ok_content = (loose.get("needs_review") == gold_nr) and all(w in loose.get("answer", "") for w in must)
        rows.append({"category": category, "question": q, "raw": raw, "format_ok": ok_fmt, "format_reason": reason,
                     "content_ok": ok_content, "both_ok": ok_fmt and ok_content, "gold_needs_review": gold_nr})
        log(f"  [{label}] {len(rows)}/{len(EVAL_QUESTIONS)}: ({category}) {q} -> {mark(ok_fmt)}/{mark(ok_content)}")
    n = len(rows)

    def rate(key):
        return {c: sum(r[key] for r in rows if r["category"] == c) / sum(1 for r in rows if r["category"] == c)
                for c in sorted({r["category"] for r in rows})}

    summary = {"label": label, "n": n,
               "format_rate": sum(r["format_ok"] for r in rows) / n,
               "content_rate": sum(r["content_ok"] for r in rows) / n,
               "both_rate": sum(r["both_ok"] for r in rows) / n,
               "format_rate_by_category": rate("format_ok"),
               "content_rate_by_category": rate("content_ok"),
               "seconds": round(time.time() - t0, 1)}
    (WORK / f"eval_{label}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary, rows


# ===== 学習前評価 =====
log("===== 学習前評価 (before) =====")
before_summary, before_rows = evaluate(model, "before")

# ===== LoRA 学習 =====
log("===== LoRA 学習（用語集データ） =====")
model.config.use_cache = False
lora = LoraConfig(r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
sft = SFTConfig(output_dir=str(WORK / "ckpt"), num_train_epochs=CONFIG["num_train_epochs"], per_device_train_batch_size=4,
                gradient_accumulation_steps=2, learning_rate=CONFIG["lr"], lr_scheduler_type="cosine",
                warmup_steps=CONFIG["warmup_steps"], logging_steps=10, save_steps=20, save_total_limit=2,
                fp16=False, max_length=CONFIG["max_seq_len"], packing=False, report_to="none", seed=CONFIG["seed"])
trainer = SFTTrainer(model=model, args=sft, train_dataset=Dataset.from_list(train), eval_dataset=Dataset.from_list(valid),
                     peft_config=lora, processing_class=tok)
trainer.model.print_trainable_parameters()
t0 = time.time()
trainer.train()
TRAIN_SECONDS = time.time() - t0
log(f"学習時間: {TRAIN_SECONDS:.0f} 秒 | ステップ数: {trainer.state.global_step} | エポック: {trainer.state.epoch:.1f}")
trainer.model.save_pretrained(WORK / "adapter")
tuned = trainer.model
tuned.config.use_cache = True
_ = tuned.eval()
loss_history = [{"step": h.get("step"), "loss": h.get("loss")} for h in trainer.state.log_history if "loss" in h]
TRAIN_LOSS_LAST = trainer.state.log_history[-1].get("train_loss") if trainer.state.log_history else None
log("loss の推移:", loss_history)
log("最終 train_loss:", TRAIN_LOSS_LAST)

# ===== 学習後評価 =====
log("===== 学習後評価 (after) =====")
after_summary, after_rows = evaluate(tuned, "after")
show(pd.DataFrame([before_summary, after_summary]).set_index("label")[["n", "format_rate", "content_rate", "both_rate", "seconds"]])
regressed = [b["question"] for b, a in zip(before_rows, after_rows) if b["both_ok"] and not a["both_ok"]]
log("退行:", regressed)
show(pd.DataFrame([{"分類": b["category"], "質問": b["question"], "正解の needs_review": b["gold_needs_review"],
                    "学習前の出力": b["raw"], "学習前 形式/内容": f"{mark(b['format_ok'])}/{mark(b['content_ok'])}",
                    "学習後の出力": a["raw"], "学習後 形式/内容": f"{mark(a['format_ok'])}/{mark(a['content_ok'])}"}
                   for b, a in zip(before_rows, after_rows)]))

# ===== 結果の保存 =====
log("===== result_homework.json 作成 =====")
result = {
    "homework_level": "発展",
    "config": {k: v for k, v in CONFIG.items() if k != "work_dir"},
    "versions": {m.__name__: m.__version__ for m in [transformers, peft, trl, torch]},
    "method": "LoRA（量子化なし）。データは scripts/make_sft_dataset.py の出力を演習と同じ JSON 形式に包んだもの＋用語集にない語の確認応答 20 件",
    "device": DEVICE,
    "n_train": len(train), "n_valid": len(valid), "n_eval": len(EVAL_QUESTIONS),
    "before": before_summary, "after": after_summary,
    "loss_history": loss_history,
    "train_loss_last": TRAIN_LOSS_LAST,
    "regressed": regressed,
    "executed_training_myself": True,
    "runtime": "Google Colab ではなくローカルマシン（DGX Spark / GB10, CUDA 13.0）で実行",
}
(WORK / "result_homework.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
log(json.dumps({k: v for k, v in result.items() if k not in ("loss_history", "config")}, ensure_ascii=False, indent=2))
log("DONE: results in", WORK)




