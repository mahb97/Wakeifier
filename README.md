# Wakeifier — Plain-to-Joyce style transfer on *Finnegans Wake*

> turn a normal sentence into something a little more Wake-ish: reproducibly, inspectably, and without hype.

**One sentence:** Wakeifier fine-tunes TinyLlama-1.1B-Chat with LoRA on *Finnegans Wake* and learns a conditional mapping  
`Plain: <normal sentence>  →  Wake: <Joycean re-rendering>`.  
It includes safe checkpoint resume on Drive, milestone adapter saves, and long-form samplers to judge “FW-ness”.

---

## Why this exists

Most style-transfer demos either (a) prompt a general chat model with examples, or (b) oversell black-box results. Wakeifier sits in the middle: small, honest, and reproducible. It shows exactly how to build a lightweight conditional objective that maps plain English to Joycean texture, with all training choices visible.

---

## Highlights

- **Conditional objective (Plain → Wake):** causal LM with label-masking so the loss only applies to the Wake side, not the conditioning prefix.
- **LoRA on attention + MLP:** `r=16, alpha=32, dropout=0.05`, target `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.
- **Tokenizer with +10 “thunderwords”** (optional) and resize for added tokens.
- **Safe resume** from the **latest valid** checkpoint on Google Drive (skips broken/incomplete ones).
- **Milestone saves** every *N* steps: `adapter_step_<n>/` + tokenizer copy.
- **Long sampler** and a simple `wakeify()` function for instant inference.
- **No chat templates** in training: pure next-token LM with a simple textual conditioning format.

---

## Setup

### Environment (Colab T4 or local)

```bash
pip install -q "transformers==4.57.1" "peft==0.12.0" "accelerate==0.34.2" "datasets==2.20.0" sentencepiece
# Torch on Colab T4 typically: 2.8.0+cu126 (preinstalled). Otherwise install matching torch+cuda build.
