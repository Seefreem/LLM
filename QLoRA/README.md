In this repo, I tried to use QLoRA to fine tune LlaMA-3B on 'xiyuez/red-dot-design-award-product-description' dataset.

Recipe:
1. Prepare data.
2. Load pretrained model with quantilization.
3. Test pretrained model before fine-tuning.
4. Config LoRA model.
5. Train LoRA model.
6. Test fine-tuned model.
7. Save adaptor.
8. Load LoRA parameters and apply fine-tuned model.