# Deep-Humor


Humor generation and classification is one the
hardest problem in the area of computational
Natural Language Understanding. Even humans
fail at being funny and recognizing humor.
In this project, we attempt to create a
joke generator using a large pre-trained language
model (GPT2). Further, we create jokes
classifier by fine-tuning pre-trained (BERT) to
classify the generated jokes and attempt to
understand what distinguish joke sentence(s)
from non joke sentence(s). Qualitative analysis
reveals that the classifier model has specific
internal attention patterns while classifying
joke sentences which is absent when classifying
normal sentences.

### Model Architecture

![Model Architecture](https://github.com/adich23/Deep-Humor/blob/master/images/model.png)

### Attention analysis

For Non-joke sentence
![Non Joke Attention](https://github.com/adich23/Deep-Humor/blob/master/images/nojoke_attention.png)


For Joke sentences, there's a visible 'X' pattern which validates the setup-punchline structure.

![Joke Attention](https://github.com/adich23/Deep-Humor/blob/master/images/joke_attention.png)

Detailed view of the Joke attention pattern - 

![detailed view](https://github.com/adich23/Deep-Humor/blob/master/images/Attentiion-explained.png)


### Software Requirements:

1. Python: 3.7
2. pytorch: 1.2.0
3. Cuda 10.1.243

Some experiments were directly carried out in Google Colab


Pretrained models are from https://github.com/huggingface/transformers

GTP2/fine_tuning.py is mostly the huggingface's run_lm_finetuning.py except for the `Dataset` Class.


### Run-GPT2-finetuning

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

### GPT2-get a sample output

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
