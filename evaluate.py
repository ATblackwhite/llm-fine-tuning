import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge
from bert_score import score
import argparse
import os

def load_model(model_path, device="cuda"):
    """加载模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=1024):
    """使用模型生成回复"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # 处理thinking内容
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)  # </think>标记
    except ValueError:
        index = 0

    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return content

def calculate_perplexity(model, tokenizer, text, max_length=1024):
    """计算困惑度"""
    encodings = tokenizer(text, return_tensors="pt").to(model.device)
    max_length = min(max_length, model.config.max_position_embeddings)

    # 只保留max_length长度
    input_ids = encodings["input_ids"][:, :max_length]
    attention_mask = encodings["attention_mask"][:, :max_length]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

    return torch.exp(loss).item()

def calculate_bleu(references, hypotheses):
    """计算BLEU分数"""
    # 对参考和假设进行分词处理
    processed_refs = [[ref.split()] for ref in references]
    processed_hyps = [hyp.split() for hyp in hypotheses]

    smoothing = SmoothingFunction().method1
    return corpus_bleu(processed_refs, processed_hyps, smoothing_function=smoothing) * 100

def calculate_rouge(references, hypotheses):
    """计算ROUGE分数"""
    rouge = Rouge()
    scores = rouge.get_scores(hypotheses, references, avg=True)
    return {
        "rouge-1": scores["rouge-1"]["f"] * 100,
        "rouge-2": scores["rouge-2"]["f"] * 100
    }

def calculate_bert_score(references, hypotheses, lang="zh"):
    """计算BERT-Score"""
    P, R, F1 = score(hypotheses, references, lang=lang, verbose=False)
    return F1.mean().item() * 100

def evaluate_model(model_path, test_data, metrics=None, output_file=None):
    """评估模型在所有指标上的表现，支持断点续跑"""
    if metrics is None:
        metrics = ["bleu", "rouge", "bert_score", "perplexity"]

    model, tokenizer = load_model(model_path)

    # 进度保存文件
    progress_file = output_file + ".tmp" if output_file else None

    # 尝试恢复进度
    references = []
    hypotheses = []
    perplexities = []
    processed_count = 0
    if progress_file and os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            progress = json.load(f)
            references = progress.get("references", [])
            hypotheses = progress.get("hypotheses", [])
            perplexities = progress.get("perplexities", [])
            processed_count = len(references)
        print(f"检测到进度文件，已恢复{processed_count}条进度，自动跳过...")

    print(f"评估模型: {model_path}")

    for idx, item in enumerate(tqdm(test_data)):
        if idx < processed_count:
            continue  # 跳过已完成
        messages = item["messages"]
        prompt = messages[0]["content"]
        reference = messages[1]["content"]
        if not reference:
            print(f"警告: 未找到助手回答，跳过此示例")
            continue
        hypothesis = generate_response(model, tokenizer, prompt)
        hypotheses.append(hypothesis)
        references.append(reference)

        # 计算困惑度
        if "perplexity" in metrics:
            perplexity = calculate_perplexity(model, tokenizer, reference)
            perplexities.append(perplexity)
        # 保存进度
        if progress_file:
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump({
                    "references": references,
                    "hypotheses": hypotheses,
                    "perplexities": perplexities
                }, f, ensure_ascii=False)

    # 过滤掉 hypothesis 为空的样本
    filtered_references = []
    filtered_hypotheses = []
    for ref, hyp in zip(references, hypotheses):
        if hyp.strip():
            filtered_references.append(ref)
            filtered_hypotheses.append(hyp)
    results = {}
    if "bleu" in metrics:
        results["bleu"] = calculate_bleu(filtered_references, filtered_hypotheses)
    if "rouge" in metrics:
        rouge_scores = calculate_rouge(filtered_references, filtered_hypotheses)
        results["rouge-1"] = rouge_scores["rouge-1"]
        results["rouge-2"] = rouge_scores["rouge-2"]
    if "bert_score" in metrics:
        results["bert_score"] = calculate_bert_score(filtered_references, filtered_hypotheses)
    if "perplexity" in metrics and perplexities:
        results["perplexity"] = np.mean(perplexities)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "model": model_path,
                "metrics": results,
                "predictions": [
                    {"prompt": item["messages"][0]["content"],
                     "reference": ref,
                     "hypothesis": hyp}
                    for item, ref, hyp in zip(test_data, references, hypotheses)
                ]
            }, f, ensure_ascii=False, indent=2)
        # 评估完成后删除进度文件
        # if progress_file and os.path.exists(progress_file):
        #     os.remove(progress_file)
    return results


def format_test_file(test_file, format_test_file):
    if format_test_file:
        formatted_data = []
        with open(test_file, 'r', encoding='utf-8') as file:
            test_data = json.load(file)
            for item in test_data:
                message_user = {
                    "role": "user",
                    "content": item["instruction"] + "\n" + item["input"]
                }
                message_assistant = {
                    "role": "assistant",
                    "content": item["output"]
                }
                formatted_item = {
                    "messages": [message_user, message_assistant]
                }
                formatted_data.append(formatted_item)
        return formatted_data
    else:
        test_data = []
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                test_data.append(item)
        return test_data

def main():
    parser = argparse.ArgumentParser(description="评估语言模型的性能")
    parser.add_argument("--models", nargs="+", required=True, help="要评估的模型路径列表")
    parser.add_argument("--test_file", required=True, help="测试数据文件")
    parser.add_argument("--format_test_file", required=False, help="是否需要对测试数据进行格式化")
    parser.add_argument("--metrics", nargs="+", default=["bleu", "rouge", "bert_score", "perplexity"],
                        help="要计算的指标列表")
    parser.add_argument("--output_dir", default="./evaluation_results", help="评估结果输出目录")
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载测试数据
    test_data = format_test_file(args.test_file, args.format_test_file)

    # 评估每个模型
    all_results = {}
    for model_path in args.models:
        model_name = model_path.split("/")[-1]
        output_file = f"{args.output_dir}/{model_name}_results.json"

        results = evaluate_model(
            model_path=model_path,
            test_data=test_data,
            metrics=args.metrics,
            output_file=output_file
        )

        all_results[model_name] = results

    # 生成比较表格
    if len(args.models) > 1:
        df = pd.DataFrame.from_dict(all_results, orient="index")
        print("\n模型比较:")
        print(df)
        df.to_csv(f"{args.output_dir}/comparison.csv")

if __name__ == "__main__":
    main()
