# Hands-on LLM Fine-Tuning with LoRA

This repository contains the hands-on material for **Session 2 of the
LLM Fine-Tuning workshop**, following the concepts introduced in Session
1.

The session takes the concepts of pretrained/instruction-tuned LLMs,
tokenization, SFT, PEFT and LoRA and turns them into a complete Google
Colab workflow.

## Workshop workflow

``` text
Environment Setup
      ↓
Dataset Preparation
      ↓
Train / Validation Split
      ↓
Tokenizer + Chat Template
      ↓
Baseline Inference
      ↓
LoRA Configuration
      ↓
Supervised Fine-Tuning
      ↓
Save / Reload Adapter
      ↓
Fine-Tuned Inference
      ↓
Base vs Fine-Tuned Evaluation
```

## Learning objectives

By completing this session you will:

-   Set up a GPU-based LLM fine-tuning environment.
-   Prepare and validate an SFT dataset.
-   Understand `system`, `user` and `assistant` roles.
-   Understand tokenization and chat templates.
-   Establish a baseline before fine-tuning.
-   Configure and train a LoRA adapter.
-   Understand the main LoRA parameters.
-   Monitor training and validation loss.
-   Save and reload a LoRA adapter.
-   Compare base and fine-tuned model outputs.
-   Understand practical LLM evaluation.
-   Decide when fine-tuning, prompting or RAG is appropriate.

## Model

The workshop starts from an **instruction-tuned Qwen causal language
model**.

The conceptual progression is:

``` text
Pretraining
    ↓
Base / pretrained model
    ↓
Instruction tuning
    ↓
Instruction-tuned model
    ↓
Our LoRA + SFT
    ↓
Specialized behavior
```

We are therefore not teaching language or instruction following from
scratch. We start with an already capable instruction-tuned model and
specialize its response behavior for a Deep Learning teaching assistant.

## Dataset

Training examples use conversational roles:

``` json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a Deep Learning teaching assistant."
    },
    {
      "role": "user",
      "content": "What is backpropagation?"
    },
    {
      "role": "assistant",
      "content": "Backpropagation is..."
    }
  ]
}
```

-   **System** --- establishes desired assistant behavior.
-   **User** --- provides the question/task.
-   **Assistant** --- provides the desired response used as the
    supervised target.

The dataset therefore teaches both **content** and **desired response
behavior**.

## Tokenization and chat templates

The structured conversation is converted into the model-specific format
using the tokenizer's chat template:

``` text
System + User + Assistant
          ↓
     Chat Template
          ↓
        Tokens
          ↓
      Token IDs
          ↓
        Model
```

The model's own chat template should be used rather than manually
inventing model-specific control tokens.

## Baseline

Before changing model parameters, generate and save responses from the
original instruction-tuned model.

Use fixed evaluation prompts such as:

-   What is backpropagation?
-   Why do neural networks need activation functions?
-   What is the difference between CNNs and Transformers?
-   Explain overfitting and dropout.
-   What is the role of the learning rate?

The same prompts are used after fine-tuning for a fair comparison.

## Why LoRA?

Full fine-tuning updates most or all model parameters and can be
expensive in memory, compute and storage.

LoRA follows the PEFT idea:

``` text
Original model
     ↓
  FROZEN
     +
Small trainable low-rank update
     ↓
Adapted model
```

The base model keeps its general capability while the adapter learns a
task-specific update.

## LoRA configuration

The workshop configuration is:

``` python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)
```

  Parameter                 Meaning
  ------------------------- --------------------------------------
  `r=8`                     Rank/capacity of the low-rank update
  `lora_alpha=16`           Scaling of the LoRA contribution
  `lora_dropout=0.05`       Regularization
  `target_modules`          Layers where LoRA is inserted
  `task_type="CAUSAL_LM"`   Causal language-modeling task
  `bias="none"`             No additional bias adaptation

In this experiment, LoRA targets the attention projection layers:

``` text
q_proj
k_proj
v_proj
o_proj
```

The original weights remain frozen; LoRA parameters receive the
gradients.

## LoRA concept

For an original weight matrix:

\[ W \]

LoRA conceptually produces:

\[ W' = W + `\Delta `{=tex}W \]

with:

\[ `\Delta `{=tex}W pprox BA \]

The original `W` is frozen while the smaller low-rank matrices are
trained.

## Supervised Fine-Tuning

SFT trains on examples where the desired response is known:

``` text
Input tokens
      ↓
Forward pass
      ↓
Token predictions / logits
      ↓
Cross-entropy loss
      ↓
Backpropagation
      ↓
Gradients
      ↓
LoRA parameter update
```

With LoRA:

``` text
Base model weights → FROZEN
LoRA parameters    → UPDATED
```

## Training configuration

The workshop example uses:

``` text
Epochs                    = 2
Per-device batch size    = 2
Gradient accumulation    = 4
Learning rate            = 2e-4
```

Effective batch size is approximately:

``` text
2 × 4 = 8
```

A generally encouraging pattern is:

``` text
Training loss ↓
Validation loss ↓
```

while:

``` text
Training loss ↓
Validation loss ↑
```

can indicate overfitting.

Loss is a diagnostic signal, not the final measure of usefulness.

## Saving and loading the adapter

Save the adapter rather than creating another full copy of the base
model:

``` python
trainer.save_model("./deep_learning_lora_adapter")
```

Reload it with:

``` python
finetuned_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)
```

Adapters can be stored and reused with the corresponding base model.

## Evaluation

Use the same prompts before and after fine-tuning:

``` text
              SAME PROMPTS
                   ↓
       ┌───────────┴───────────┐
       ↓                       ↓
   BASE MODEL             FINE-TUNED
       ↓                       ↓
    Answers                 Answers
       └───────────┬───────────┘
                   ↓
                Compare
```

Evaluate:

-   Correctness
-   Relevance
-   Clarity
-   Conciseness
-   Terminology
-   Teaching style
-   Consistency

A different answer is not automatically a better answer. The goal is
**desired behavior improvement**, not simply output change.

### LLM evaluation methods

-   **Training metrics:** training loss, validation loss, perplexity.
-   **Task metrics:** accuracy, F1, exact match when ground truth is
    well-defined.
-   **Qualitative evaluation:** correctness, relevance, clarity,
    consistency and style.
-   **Human evaluation:** especially useful for open-ended generation.
-   **LLM-as-a-judge:** scalable, but should be checked for judge bias
    against human evaluation.

## Practical decision: prompting, fine-tuning or RAG?

### Prompt engineering

Change instructions/context at inference time. No training required.

### Prompt tuning

Learn continuous prompt parameters while keeping the model largely
frozen.

### Fine-tuning

Adapt behavior, task execution or response style through training.

### RAG

Retrieve external information and provide it to the model at inference
time. Particularly useful for changing or proprietary knowledge.

Fine-tuning and RAG can also be combined.

## Reproducibility checklist

-   Record model name/version.
-   Record Python and library versions.
-   Record GPU/runtime information.
-   Keep dataset and train/validation split fixed.
-   Save baseline responses.
-   Record LoRA configuration.
-   Record training hyperparameters.
-   Save the adapter.
-   Reuse the same evaluation prompts.
-   Use consistent evaluation criteria.

## Repository structure

A recommended structure is:

``` text
.
├── README.md
├── notebooks/
│   └── session2_llm_fine_tuning.ipynb
├── data/
│   ├── train.json
│   └── validation.json
├── scripts/
│   ├── setup.py
│   ├── dataset_prep.py
│   ├── baseline.py
│   ├── lora_config.py
│   ├── train.py
│   ├── save_load.py
│   └── compare.py
└── outputs/
    └── deep_learning_lora_adapter/
```

Adapt this structure to the files actually committed to the repository.

## Takeaway

``` text
Pretrained capability
        ↓
Instruction-tuned model
        ↓
Curated SFT dataset
        ↓
LoRA adapter
        ↓
Specialized behavior
        ↓
Evaluation
```

> **Don't just fine-tune a model. Fine-tune it for a reason --- and
> measure whether it worked.**
