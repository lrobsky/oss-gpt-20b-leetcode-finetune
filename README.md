# oss-gpt-20b-leetcode-finetune
This repository contains the code and training workflow for a finetuned version of [oss-gpt-20b](https://huggingface.co/unsloth/gpt-oss-20b), specialized for solving LeetCode-style coding problems.

The model was finetuned using [Unsloth](https://github.com/unslothai/unsloth) on the [LeetCode Dataset](https://huggingface.co/datasets/newfacade/LeetCodeDataset).

## Notes  
- The model weights are not stored in this GitHub repo due to size limits.  
  You can find them on Hugging Face through the links below.
- Currently, this model cannot be exported to GGUF.
  For inference outside Python, you will need GGUF support, which is not available for oss-gpt-20b QLoRA adapters using Unsloth yet.

---

## Repository Contents

- `gpt_oss_(20B)_Fine_tuning.ipynb` – notebook used for finetuning with Unsloth.  
- `inference_example.py` – example script showing how to load the model from Hugging Face and run it on a LeetCode problem.  
- `requirements.txt` – dependencies for local inference. 

---

## Quickstart

Clone this repository and install dependencies:

```bash
git clone https://github.com/lrobsky/oss-gpt-20b-leetcode-finetune.git
cd oss-gpt-20b-leetcode-finetune
pip install -r requirements.txt

```






## Model Weights

The model weights are hosted on Hugging Face in multiple formats:

- **Full merged model** – plug-and-play version, no adapters required.   👉 [View the Hugging Face page](https://huggingface.co/lrobsky/gpt-oss-20b-finetuned-leetcode-merged)
- **Adapter weights (QLoRA)** – lightweight, to be applied on top of the base `oss-gpt-20b`.  👉 [View the Hugging Face page](https://huggingface.co/lrobsky/gpt-oss-20b-finetuned-leetcode)
- **GGUF quantized versions** – for use with [llama.cpp](https://github.com/ggerganov/llama.cpp), or [Ollama](https://github.com/ollama/ollama).  **TO BE UPDATED (once available)**



---

## License

This repository follows the same license as the base model ([oss-gpt-20b](https://huggingface.co/openai/gpt-oss-20b)), unless otherwise specified.

---

## References

- [OSS-GPT-20B Base Model](https://huggingface.co/openai/gpt-oss-20b)  
- [LeetCode Dataset](https://huggingface.co/datasets/newfacade/LeetCodeDataset)  
- [Unsloth](https://github.com/unslothai/unsloth)  
- [PEFT / LoRA](https://huggingface.co/docs/peft)  
