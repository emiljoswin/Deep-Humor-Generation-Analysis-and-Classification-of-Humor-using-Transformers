# Deep-Humor

URL: https://github.com/adich23/Deep-Humor

Software Requirements:

1. Python: 3.7
2. pytorch: 1.2.0
3. Cuda 10.1.243

Some experiments were directly carried out in Google Colab


Pretrained models are from https://github.com/huggingface/transformers

GTP2/fine_tuning.py is mostly the huggingface's run_lm_finetuning.py except for the `Dataset` Class.


#### Run-GPT2-finetuning

```
python fine_tuning.py \
	--output_dir output \
	--model_type gpt2 \
	--model_name_or_path distilgpt2 \
	--do_train \
	--train_data_file short_jokes_even_shorter.csv \
	--per_gpu_train_batch_size 5 \
	--save_steps 1000 \
	--num_train_epochs 10
```

#### GPT2-get a sample output

```
python run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path=output/ \
    --top_k 50 \
    --top_p 1.0 \
    --temperature 0.3 \
    --prompt "Why did jonh call the cops?"
```

An improved version is in the jupyter notebook along with the rest of the code for generation, analysis and classification.

### BERT

Jupyter notebook is run on Google Collab, any extra package requried required is being included in the notebook itself.
