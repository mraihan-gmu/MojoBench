
<div align="center">
<h1>🔥 Mojo-Coder 🔥</h1>
<em>State-of-the-art Language Model for Mojo Programming</em>
</div>


<div align="center">
<table><tr>
<td><a href="https://arxiv.org/abs/2410.17736"><img src="https://img.shields.io/badge/arXiv-Read_Paper-blue?style=for-the-badge&logo=arxiv" /></a></td>
<td><a href="mailto:mraihan2@gmu.edu"><img src="https://img.shields.io/badge/Email-Contact_Us-blue?style=for-the-badge&logo=gmail" /></a></td>
</tr></table>
</div>



<div align="center">
<h2>🎯 Background and Motivation</h2>
</div>

Mojo programming language, developed by Modular, has emerged as a game-changing technology in high-performance computing and AI development. Despite its growing popularity and impressive capabilities (up to 68,000x faster than Python!), existing LLMs struggle with Mojo code generation. Mojo-Coder addresses this gap by providing specialized support for Mojo programming, built upon the robust architecture of [CodeGemma-7B-IT](https://huggingface.co/google/codegemma-7b-it/).

<div align="center">
<h2>🤖 Model Information</h2>
</div>

Mojo-Coder transforms natural language instructions into optimized Mojo code, supporting multiple languages (English, German, French, Spanish, and Bangla) while maintaining high-quality code generation capabilities.

<div align="center">
<h2>📝 Description</h2>
</div>

The Mojo-Coder family consists of three specialized 7B-parameter models, each built on CodeGemma's architecture:
|                            | <h3><a href="https://huggingface.co/md-nishat-008/mojo-coder" style="color: #0969DA;">mojo-coder</a> 🔥</h3> | <h3><a href="https://huggingface.co/md-nishat-008/mojo-coder-it" style="color: #0969DA;">mojo-coder-it</a> 🎆</h3> | <h3><a href="https://huggingface.co/md-nishat-008/mojo-coder-it-m" style="color: #0969DA;">mojo-coder-it-m</a> ⭐</h3> |
|---------------------------|:---:|:---:|:---:|
| 🔄 Code Completion        | ✅ | ✅ | ✅ |
| 💡 NL → Code Generation   |    | ✅ | ✅ |
| 🌏 Multilingual Support   |    |    | ✅ |
| 📝 Instruction Following  |    | ✅ | ✅ |

<div align="center">
<h2>🚀 Sample Usage</h2>
</div>

Choose the model that best fits your needs:
- For basic Mojo code completion: [mojo-coder](https://huggingface.co/md-nishat-008/mojo-coder)
- For English instruction-based code generation: [mojo-coder-it](https://huggingface.co/md-nishat-008/mojo-coder-it)
- For multilingual support: [mojo-coder-it-m](https://huggingface.co/md-nishat-008/mojo-coder-it-m)

Notably, our models significantly outperform current state-of-the-art models including GPT-4o and Claude-3.5-Sonnet on the HumanEval-Mojo benchmark.


<div style="color: red; text-align: center; padding: 10px; margin: 20px 0; border: 2px solid red; border-radius: 5px;">
<strong>⚠️ IMPORTANT: When using the model, you MUST explicitly mention "Mojo" in your prompts (e.g., "Write a Mojo function to...", "Create Mojo code that...") otherwise the model may not generate Mojo code!</strong>
</div>

#### For Code Generation

```python
from transformers import GemmaTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("md-nishat-008/Mojo-Coder-it")
model = AutoModelForCausalLM.from_pretrained("md-nishat-008/Mojo-Coder-it")

input_text = "Write me a Mojo function to calculate the nth fibonacci number."
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
```

#### Chat Template

The instruction-tuned models use a chat template that must be adhered to for conversational use.
The easiest way to apply it is using the tokenizer's built-in chat template, as shown in the following snippet.

Let's load the model and apply the chat template to a conversation. In this example, we'll start with a single user interaction:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("md-nishat-008/Mojo-Coder-it")
model = AutoModelForCausalLM.from_pretrained("md-nishat-008/Mojo-Coder-it")

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

```py
inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)
```

<div align="center">
<h2>⚙️ Inputs and Outputs</h2>
</div>

**Inputs**:
- For base model (mojo-coder): code prefix and/or suffix for Mojo code completion
- For instruction-tuned models (mojo-coder-it & mojo-coder-it-m): natural language prompts/instructions

<p style="color: red;"><strong>Note: In prompts, you must explicitly mention "Mojo" (e.g., "Write a Mojo function to...", "Write Mojo code to...") otherwise the models may not generate Mojo code.</strong></p>

**Outputs**:
- For all variants: Mojo code snippets and natural language responses
- Additional explanations and documentation when requested

<div align="center">
<h2>📚 Model Data</h2>
</div>

### Training Dataset

Using [CodeGemma-7B-IT](https://huggingface.co/google/codegemma-7b-it/) as our base model, we further trained on:
- [Mojo-Corpus](https://huggingface.co/datasets/md-nishat-008/Mojo_Corpus): 6.5M tokens of curated Mojo code from public repositories
- [Mojo-SFT](https://huggingface.co/datasets/md-nishat-008/Mojo_SFT): 3,200 instruction-code pairs for English
- [Mojo-mSFT](https://huggingface.co/datasets/md-nishat-008/Mojo_mSFT): Multilingual instruction-code pairs in 5 languages

### Training Data Processing

The following data pre-processing techniques were applied:
- Rigorous filtering pipeline (F1-F6) to ensure code quality
- Apache 2.0 license compliance
- Language detection using fastText
- Duplicate removal and content validation
- Expert review for instruction-code pairs

<div align="center">
<h2>📊 Evaluation Information</h2>
</div>

### Evaluation Approach

We evaluate Mojo-Coder on:
- [HumanEval-Mojo](https://huggingface.co/datasets/md-nishat-008/HumanEval-Mojo): First benchmark for Mojo code generation
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

<div align="center">
<h2>⚠️ Limitations and Usage</h2>
</div>

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



<div align="center">
<h2>📚 Citation</h2>
</div>

If you find our work helpful, please consider citing our paper:

<div style="background-color: #f6f8fa; padding: 20px; border-radius: 5px; margin: 10px 0;">
<p style="margin-bottom: 10px;"><strong>MojoBench: Language Modeling and Benchmarks for Mojo</strong></p>

```bibtex
@inproceedings{Raihan2024MojoBenchLM,
   title     = {MojoBench: Language Modeling and Benchmarks for Mojo},
   author    = {Raihan, Nishat and Santos, Joanna C. S. and Zampieri, Marcos},
   year      = {2024},
   url       = {https://api.semanticscholar.org/CorpusID:273532552}
}
```
