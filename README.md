# AIDC-W3-LABS
repo for all the labs done in week 3


# Week 3 Labs: Prediction Cards

A tracking log of predictions and benchmark calculations across Week 3 lab sessions covering LLM memory footprint, KV cache sizing, and serving throughput.

---

## Day 1 (w3d1): Memory Footprints & GPU Utilization

| # | Question / Prompt | Prediction |
| :---: | :--- | :--- |
| **01** | Qwen2.5-1.5B-Instruct is about 1.5 billion parameters. At fp16 (2 bytes each) the weights alone are about `______` GB. At int8 (1 byte each) about `______` GB. | `3.0GB, 1.5GB` |
| **02** | Resident VRAM at 512 context, fp16: `______` GB. At 4096 context, fp16: `______` GB. *(Which is larger, and by roughly how much?)* | `3.0GB, 4.5GB` |
| **03** | During a single-request decode (one prompt, generating tokens one at a time), GPU utilisation will read about `______` percent. | `90` |

---

## Day 2 (w3d2): Prefill, Decode & KV Cache

| # | Question / Prompt | Prediction |
| :---: | :--- | :--- |
| **01** | Time to first token (TTFT) is dominated by prefill (reading the whole prompt). A longer prompt makes TTFT go `______` *(up / down / no change)*. | `no change` |
| **02** | After the first token, decode emits one token at a time. The mean gap between tokens (TPOT) depends mostly on `______` *(prompt length / model size and memory bandwidth)*. | `prompt length` |
| **03** | KV cache math for Qwen2.5-1.5B: 28 layers, 2 KV heads, head_dim 128, fp16. Per token that is $2 \times 28 \times 2 \times 128 \times 2\text{ bytes} =$ `______` KB per token. | `32KB` |
| **04** | So a 4096-token context holds about `______` GB of KV. | `1GB` |
| **05** | Static batching: if you pad 8 prompts of different lengths and run them as one batch, the batch finishes when the `______` prompt finishes. | `longest output` |

---

## Day 3 (w3d3): Continuous Batching & Scaling

| # | Question / Prompt | Prediction |
| :---: | :--- | :--- |
| **01** | At concurrency 8, vLLM's throughput (tokens/s across all requests) will be about `______` times Monday's static-batch-8 number. Write the multiple down. | `4x` |
| **02** | vLLM runs the identical queue. Predict how far IT scales from concurrency 1 to 8: `______` x. *(Calculated below)* | `2.9x` |
| **03** | Monday's `slot_efficiency` collapsed to about a third once the queue had mixed output lengths. Continuous batching does not pay that tax. So you should expect vLLM's scaling multiple to be `______` *(larger / smaller / the same as)* static batching's, and roughly `______` x larger. | `larger, 2` |

### Baselines Reference & Day 3 Card 2 Calculation

```json
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "dtype": "fp16",
  "ttft_s": {
    "128": 0.1432,
    "512": 0.0638,
    "2048": 0.2962
  },
  "tpot_s": 0.0388,
  "batch": {
    "1": 34.9,
    "4": 51.6,
    "8": 100.0
  }
}
