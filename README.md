# 🔥 Mojo-Coder 🔥

*State-of-the-art Language Model for Mojo Programming*

## 🎯 Background and Motivation

Mojo programming language, developed by Modular, has emerged as a game-changing technology in high-performance computing and AI development. Despite its growing popularity and impressive capabilities (up to 68,000x faster than Python!), existing LLMs struggle with Mojo code generation. Mojo-Coder addresses this gap by providing specialized support for Mojo programming, built upon the robust architecture of CodeGemma-7B-IT.

## 🤖 Model Information

Mojo-Coder transforms natural language instructions into optimized Mojo code, supporting multiple languages (English, German, French, Spanish, and Bangla) while maintaining high-quality code generation capabilities.

## 📝 Description

The Mojo-Coder family consists of three specialized 7B-parameter models, each built on CodeGemma's architecture:

|                            | mojo-coder 🔥 | mojo-coder-it 🎆 | mojo-coder-it-m ⭐ |
|---------------------------|:---:|:---:|:---:|
| 🔄 Code Completion        | ✅ | ✅ | ✅ |
| 💡 NL → Code Generation   |    | ✅ | ✅ |
| 🌏 Multilingual Support   |    |    | ✅ |
| 📝 Instruction Following  |    | ✅ | ✅ |

## 🚀 Sample Usage

Choose the model that best fits your needs:
- For basic Mojo code completion: mojo-coder
- For English instruction-based code generation: mojo-coder-it  
- For multilingual support: mojo-coder-it-m

Notably, our models significantly outperform current state-of-the-art models including GPT-4o and Claude-3.5-Sonnet on the HumanEval-Mojo benchmark.

> ⚠️ **IMPORTANT**: When using the model, you MUST explicitly mention "Mojo" in your prompts (e.g., "Write a Mojo function to...", "Create Mojo code that...") otherwise the model may not generate Mojo code!

### For Code Generation

```python
from transformers import GemmaTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("********")
model = AutoModelForCausalLM.from_pretrained("*********")

input_text = "Write me a Mojo function to calculate the nth fibonacci number."
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
```

### Chat Template

The instruction-tuned models use a chat template that must be adhered to for conversational use.
The easiest way to apply it is using the tokenizer's built-in chat template, as shown in the following snippet:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("********")
model = AutoModelForCausalLM.from_pretrained("*********")

chat = [{"role": "user", "content": "Write a function that calculates factorial of a number in Mojo"}]
inputs = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        inputs=inputs,
        max_new_tokens=1000,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

At this point, the prompt contains the following text:

```
<bos><start_of_turn>user
Write a hello world program in Mojo<end_of_turn>
<start_of_turn>model
```

As you can see, each turn is preceded by a `<start_of_turn>` delimiter and then the role of the entity
(either `user`, for content supplied by the user, or `model` for LLM responses). Turns finish with
the `<end_of_turn>` token.

You can follow this format to build the prompt manually, if you need to do it without the tokenizer's
chat template.

After the prompt is ready, generation can be performed like this:

```python
inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)
```

## ⚙️ Inputs and Outputs

**Inputs**:
- For base model (mojo-coder): code prefix and/or suffix for Mojo code completion
- For instruction-tuned models (mojo-coder-it & mojo-coder-it-m): natural language prompts/instructions

**Note**: In prompts, you must explicitly mention "Mojo" (e.g., "Write a Mojo function to...", "Write Mojo code to...") otherwise the models may not generate Mojo code.

**Outputs**:
- For all variants: Mojo code snippets and natural language responses
- Additional explanations and documentation when requested

## 📚 Model Data

### Training Dataset

Using CodeGemma-7B-IT as our base model, we further trained on:
- Mojo-Corpus: 6.5M tokens of curated Mojo code from public repositories
- Mojo-SFT: 3,200 instruction-code pairs for English
- Mojo-mSFT: Multilingual instruction-code pairs in 5 languages

### Training Data Processing

The following data pre-processing techniques were applied:
- Rigorous filtering pipeline (F1-F6) to ensure code quality
- Apache 2.0 license compliance
- Language detection using fastText
- Duplicate removal and content validation
- Expert review for instruction-code pairs

## 📊 Evaluation Information

### Evaluation Approach

We evaluate Mojo-Coder on:
- HumanEval-Mojo: First benchmark for Mojo code generation
- Multi-language instruction following
- Code quality and execution success

### Evaluation Results

#### Code Generation Benchmarks (Pass@1)

| Model | HumanEval-Mojo |
|-------|----------------|
| GPT-4o | 25.5% |
| Claude-3.5-Sonnet | 39.8% |
| mojo-coder | 36.7% |
| mojo-coder-it-m | 61.5% |
| mojo-coder-it | 66.4% |

## ⚠️ Limitations and Usage

### Intended Usage
- Mojo code completion and generation
- Multi-language instruction following
- Code documentation and explanation
- Educational support for Mojo programming

### Known Limitations
- Limited to Mojo programming language
- Requires explicit mention of "Mojo" in prompts
- Performance may vary with complex algorithms
- May occasionally generate Python-like syntax
- Based on data available up to 2024

### Ethical Considerations
The model is designed for:
- Educational and development purposes
- Open-source contribution to Mojo ecosystem
- Supporting multilingual access to Mojo programming

Code should be reviewed and tested before production use, especially for performance-critical applications.
