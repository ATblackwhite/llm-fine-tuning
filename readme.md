# fine-tuning for Qwen3-4B&Qwen3-4B-Base

## Env Preparation

```shell
modelscope download --model Qwen/Qwen3-4B-Base --local_dir ./Qwen3-4B-Base
modelscope download --model Qwen/Qwen3-4B --local_dir ./Qwen3-4B

```

## Fine tuning

```shell
llamafactory-cli train ./qwen3_4b_base_lora_sft.yaml
llamafactory-cli train ./qwen3_4b_lora_sft.yaml

```

## Evaluation

```shell
llamafactory-cli train ./qwen3_4b_base_lora_sft_eval.yaml
llamafactory-cli train ./qwen3_4b_lora_sft_eval.yaml

python evaluate.py --models ./Qwen3-4B-Base --test_file data/test_data.json --format_test_file true
```
