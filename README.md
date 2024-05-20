# Collaiborator-MEDLLM-Llama-3-8B-v1 

![image/png](https://cdn-uploads.huggingface.co/production/uploads/653f5b93cd52f288490edc83/wIES_YhNPKn--AqcEmzRJ.png)

This model is a fine-tuned version of [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) on our custom "BioMedData" dataset.

## Model details

Model Name: Collaiborator-MEDLLM-Llama-3-8b

Base Model: Llama-3-8B-Instruct

Parameter Count: 8 billion

Training Data: Custom high-quality biomedical dataset

Number of Entries in Dataset: 500,000+

Dataset Composition: The dataset comprises both synthetic and manually curated samples, ensuring a diverse and comprehensive coverage of biomedical knowledge.

Training Hardware: NVIDIA A40 GPU

## Model description

Collaiborator-MEDLLM-Llama-3-8b is a specialized large language model designed for biomedical applications. It is finetuned from the Llama-3-8B-Instruct model using a custom dataset containing over 500,000 diverse entries. These entries include a mix of synthetic and manually curated data, ensuring high quality and broad coverage of biomedical topics.

The model is trained to understand and generate text related to various biomedical fields, making it a valuable tool for researchers, clinicians, and other professionals in the biomedical domain.

## Quick Demo

<video controls autoplay src="https://cdn-uploads.huggingface.co/production/uploads/653f5b93cd52f288490edc83/piGRPwvcBTLmcgExL89zp.mp4"></video>

## Intended uses & limitations

Collaiborator-MEDLLM-Llama-3-8b is intended for a wide range of applications within the biomedical field, including:

1. Research Support: Assisting researchers in literature review and data extraction from biomedical texts.
2. Clinical Decision Support: Providing information to support clinical decision-making processes.
3. Educational Tool: Serving as a resource for medical students and professionals seeking to expand their knowledge base.

## Limitations and Ethical Considerations

While Collaiborator-MEDLLM-Llama-3-8b performs well in various biomedical NLP tasks, users should be aware of the following limitations:

Biases: The model may inherit biases present in the training data. Efforts have been made to curate a balanced dataset, but some biases may persist.

Accuracy: The model's responses are based on patterns in the data it has seen and may not always be accurate or up-to-date. Users should verify critical information from reliable sources.

Ethical Use: The model should be used responsibly, particularly in clinical settings where the stakes are high. It should complement, not replace, professional judgment and expertise.


## Training and evaluation

Collaiborator-MEDLLM-Llama-3-8b was trained using an NVIDIA A40 GPU, which provides the computational power necessary for handling large-scale data and model parameters efficiently. Rigorous evaluation protocols have been implemented to benchmark its performance against similar models, ensuring its robustness and reliability in real-world applications.

## How to use

import transformers
import torch

model_id = "SrikanthChellappa/Collaiborator-MEDLLM-Llama-3-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are an expert trained on healthcare and biomedical domain!"},
    {"role": "user", "content": "I'm a 35-year-old male and for the past few months, I've been experiencing fatigue, increased sensitivity to cold, and dry, itchy skin. What is the diagnosis here?"},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])

### Contact Information

For further information, inquiries, or issues related to Biomed-LLM, please contact:

Email: info@collaiborate.com

Website: https://www.collaiborate.com

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.03
- training_steps: 2000
- mixed_precision_training: Native AMP

### Framework versions

- PEFT 0.11.0
- Transformers 4.40.2
- Pytorch 2.1.2
- Datasets 2.19.1
- Tokenizers 0.19.1

### Citation

If you use Collaiborator-MEDLLM-Llama-3-8b in your research or applications, please cite it as follows:

@misc{Collaiborator_MEDLLM,
  author = Collaiborator,
  title = {Collaiborator-MEDLLM-Llama-3-8b: A High-Performance Biomedical Language Model},
  year = {2024},
  howpublished = {https://huggingface.co/collaiborateorg/Collaiborator-MEDLLM-Llama-3-8B-v1},
}
