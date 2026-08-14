# Routing harness

Experiment harness and provenance gate that produce cost-aware routing cells. Integrity receipts are not a routing win.

**[Site](https://peterponyu.github.io/cost-aware-edit-routing-harness/)** · **[Source](https://github.com/PeterPonyu/cost-aware-edit-routing-harness)**

# Vochn Wisdom Frontier ideas — local-hardware feasibility analysis

> Generated: 2026-06-30 · Analysis machine: local machine (actually measured, not a memory-based guess)
> One-line conclusion: **The 24 GB single GPU is the only hard constraint.** Any idea that is "algorithm / post-training / fine-tuning a small-to-medium model" can almost all be done on this machine; any idea that requires "pretraining a large model from scratch / large-scale video world models / real hardware" cannot be done on this machine (or can only be done as inference / evaluation / a scaled-down version).

---

## 0. Measured hardware (verified with `nvidia-smi` / `torch`)

| Component | Measured value | Implication for research |
|---|---|---|
| GPU | **RTX 5090 Laptop, 24 GB VRAM**, Blackwell CC 12.0, driver 580 | A single 24 GB GPU is the ceiling. No NVLink, no multi-GPU. |
| CPU | Intel Core Ultra 9 275HX, **24 cores** | Enough for data preprocessing / classical ML / simulation. |
| RAM | **62 GB** (+8 GB swap, 5.7 GB already used) | CPU offload is possible, but swap headroom is already tight. |
| Disk | 1.9 TB total, **579 GB free** | Large datasets / multiple model weights need to be used sparingly (HF+Ollama history already takes ~380 GB). |
| Software stack | torch **2.12.1+cu130**, transformers 5.12.1, trl 0.24.0 | The LLM fine-tuning chain is ready. |

### Environment issue that needs immediate fixing
The **vision stack in the `dl` environment is currently broken**:
```
RuntimeError: operator torchvision::nms does not exist   # torchvision/torchao mismatch with torch 2.12.1
```
This also brings down `peft` / `diffusers` / any object-detection import along with it. **Any vision/diffusion idea must fix this before starting work** (reinstall a torchvision matching torch 2.12.1+cu130, ~10 minutes). Pure text LLM fine-tuning is unaffected.

---

## 1. The 24 GB capability envelope (keep these numbers in mind — every verdict below is derived from them)

| Task type | Feasible with 24 GB | Specific boundary |
|---|---|---|
| LLM **inference** | Yes | Up to 14B natively in fp16; 32B can be inferred with 4-bit quantization; 70B only with extreme quantization / Ollama offload. |
| LLM **LoRA/QLoRA fine-tuning** | Yes for ≤14B comfortably / marginal for 32B QLoRA | 7–8B QLoRA is very comfortable (already have 5,339 SFT-run experience); 14B QLoRA is fine; 32B QLoRA needs gradient checkpointing + small batch size — runs but slow. 70B training is not possible. |
| **Full pretraining** of a large model | No | Pretraining any model ≥1B from scratch is unrealistic; requires a multi-GPU cluster. |
| **Embedding / reranker training** | Yes | Already verified (14 embedding models + 6 reranker models running). |
| **Diffusion image** LoRA/fine-tuning | Limited | SD1.5/SDXL LoRA: yes; FLUX/large-DiT LoRA: marginal (needs quantization + offload). |
| **Video generation world model** training | No / partial | Full training needs multiple GPUs. Only feasible: inference, distilled "student-side" small models, training a policy on pre-extracted features. |
| **VLA / embodied** (algorithm side) | Limited | Training a policy on **offline data / pre-extracted features** is fine; **real-robot execution / online rollout is not possible** (no robotic arm). |
| **Object detection / segmentation / classification** | Yes | Fine-tuning ResNet/ViT/DETR/D-FINE/SAM is all fine (fix torchvision first). |
| **Time series / recommendation / GNN / federated learning** | Yes | Medium scale is all fine. |
| **Materials / chemistry / mechanical** | No (GPU-irrelevant) | Requires a lab, a CFD cluster, or dedicated equipment — unrelated to this machine. |

---

## 2. Feasibility verdicts for all ideas (clustered by resource profile, not listed one by one)

Legend: Yes = directly feasible on this machine · Limited = feasible with constraints (quantization/scaled-down/small batch) · Partial = inference/evaluation/scaled-down reproduction only on this machine · No = not feasible on this machine (lacks real hardware/multi-GPU/specialized data)

### Category A (Yes) — this machine's sweet spot (strongly recommended, easiest to produce results)
> These are essentially all "algorithm/post-training/evaluation work on small-to-medium models" — 24 GB is more than enough.

| Idea cluster | Specific topics involved | Notes |
|---|---|---|
| **Knowledge editing / lifelong editing** (a whole CCF-A group) | Generalizable lifelong editing, mixing in-model and external knowledge, long-context editing, multi-fact editing, Hopfield memory routing, multimodal editing, RL editing policies, pruning closed-loop, editing theory, stopping criteria | Almost all done on ≤7B models — the sweetest direction for this machine, one group of 9 topics. |
| **Efficient reasoning / CoT** | LLM/MLLM CoT efficiency, reasoning token compression | 7–14B inference plus light fine-tuning suffices. |
| **Post-training / alignment-RL** | RL-based post-training alignment, RL-based large models, RL fine-tuning | DPO/GRPO run directly with trl on 7–8B models. |
| **Agent / multi-agent / agent safety** | Agent architectures, multi-agent systems, LLM/agent safety, LLM long-term memory mechanisms | Mostly orchestration + evaluation + small-model fine-tuning — this machine is sufficient. |
| **Multimodal knowledge-graph reasoning** | Multimodal KG + LLM reasoning | Retrieval + reasoning — this machine is sufficient. |
| **Retrieval / recommendation / time series** | LLM-based recommendation, top-N recommendation, LLM+Transformer time-series forecasting, multimodal fusion for AIGC detection | Shares the same lineage as the existing scMetaIntel retrieval stack. |
| **Interpretability / data augmentation** | Deep-model interpretability, data augmentation under data scarcity, AutoML | Low bar for SCI Q3, easy on this machine. |
| **Graph learning / federated / SNN** | Federated multimodal learning, imbalance-robust graph learning, spiking neural networks for CV | Medium scale is feasible on this machine. |
| **Discriminative vision** (after fixing torchvision) | Object detection, fine-grained classification, sign-language/action recognition, industrial object detection, multi-object tracking, 3D weakly-supervised segmentation, remote-sensing interpretation/fusion/change-detection, root-cause diagnosis | Mostly SCI Q3–Q4, standard configuration on this machine works. |

### Category B (Limited) — feasible on this machine with constraints (quantization / scaled-down / offline data)
| Idea cluster | Specific topics | Constraint |
|---|---|---|
| **Unified image understanding+generation / image editing** | Unified understanding+generation post-training+inference acceleration, human-instruction-aligned image editing, controllable image generation and editing | Full training of a unified large model: no; LoRA/post-training on an open-source backbone: limited but feasible. Fix torchvision first. |
| **Diffusion-based embodiment / VLA (algorithm side)** | Diffusion-based robotic-arm manipulation understanding and generation, VLA discrete diffusion, 3D-ViT VLA spatial reasoning, action-associated world modeling, executable future-representation learning | Training a policy **on offline demo data / pre-extracted features** is feasible; real-robot/online rollout is not. See the dedicated "world-model distillation" section below. |
| **Edge MoE / video understanding** | On-demand edge MoE expert scheduling, video analysis for understanding complex real-world scenes, video spatial-perception understanding, video token compression | Student/small-model side is feasible on this machine; training a large MoE is not. |

### Category C (Partial) — this machine can only do inference / evaluation / scaled-down reproduction
| Idea cluster | Specific topics | Why |
|---|---|---|
| **Video generation world models** | Open-space geometry-consistent world-model video generation, long-video narrative generation, video special-effects foundation models, identity-preserving face video, efficient video-generation distillation and real-time inference | Full training requires an 8×A100-class cluster. This machine can only: run inference on open-source models, do the **student side of distillation**, do evaluation/analysis. |
| **3D reconstruction / Gaussian splatting** | Adversarial attacks on VGGT feed-forward 3D reconstruction, adversarial examples on 3D Gaussians, lightweight feed-forward 3D reconstruction, non-coplanar sound-source 3D localization | Feasible (partial) for adversarial/lightweight work on small scenes/pretrained models; training a large reconstruction model from scratch is not. |
| **Neuroscience neural decoding** | Neural signal decoding + adaptation | The model itself is fine on this machine, but it **requires EEG/neural data** — that is the data barrier. |

### Category D (No) — not feasible on this machine (not a GPU-capacity issue, but lack of real hardware / specialized platform)
| Idea cluster | Specific topics | What's missing |
|---|---|---|
| **Embodied/VLA execution requiring real hardware** | "Train with World Models, Execute without" (the proposer already notes **real hardware required**), VLA robotic-arm grasping on real hardware, full-scenario embodied robots, robot navigation/localization on special terrain | **A robotic arm / physical robot.** A simulation version can be downgraded to Category B/C; real-hardware evaluation is not possible. |
| **Autonomous driving (real closed-loop)** | Obstacle avoidance / human-machine interaction, hazardous-object detection in special weather, vehicular-network edge offloading | The detection part can be done with public datasets (→ Category A); **real-vehicle closed-loop / vehicular-network field testing is not possible.** |
| **Mechanical engineering / physical experiments** | Oil-film dynamics of sliding bearings in wind-turbine gearboxes | A CFD cluster / test-bench experiments. |
| **Humanities/social sciences** | Digital capability and elderly asset allocation | Not a computational topic. |

---

## 3. Dedicated verdict for the world-model distillation direction (the two English-titled topics the proposer specifically asked about)

> "Train with World Models, Execute without World Models" — distill future-prediction ability into the action policy: train with future supervision, output actions directly at inference time.

| Sub-direction | Verdict for this machine |
|---|---|
| **Value-of-Imagination Distillation without Future Rendering** | Limited → Yes, **the most feasible world-model topic on this machine.** The core innovation — "use future supervision during training, don't render during inference" — precisely avoids the most expensive part on this machine, video generation. Training a policy on an **offline dataset (e.g., a RoboMimic / RT-X subset / open-source manipulation demos)** fits within 24 GB. |
| **Decision-Influential World Abstraction for VLA-WAM** | Limited, same as above — the algorithm/representation side is feasible on this machine; be sure to use **pre-extracted features** rather than running a large WAM end-to-end. |
| **Real-hardware execution evaluation** | No — the proposer already notes "requires real hardware" — this step cannot be done on this machine / without a robotic arm. **Workaround: substitute a simulation benchmark (e.g., LIBERO / ManiSkill / CALVIN) for real-hardware evaluation**, leave "real hardware" as future work — the paper can still be submitted. |

**This line is directly homologous to the `robotics-embodied-reliability-research` repo you already have on the Desktop** — one of the fastest directions to produce results by reusing existing assets.

---

## 4. Recommended shortlist: feasible on this machine AND fastest to produce results (combined with your existing assets)

> You already have 10+ reliability/research repos on the Desktop and 4 already-validated publishable directions. The list below prioritizes the intersection of "algorithmic sweet spot + existing assets."

1. **The whole knowledge-editing group (CCF-A)** — 9 sub-topics, pure ≤7B algorithm work, no data barrier, the sweetest spot on this machine. **First choice for building skill + a coherent series.**
2. **World-model distillation "Value-of-Imagination without Future Rendering"** — reuse `robotics-embodied-reliability-research`, substitute simulation evaluation for real hardware, CCF-A caliber.
3. **Efficient reasoning / CoT + post-training RL** — runs directly with trl on 7–8B, reuses existing fine-tuning experience (unsloth+LoRA).
4. **Retrieval/RAG/recommendation/time-series family** — directly reuses the embedding+reranker+RAG stack from `scMetaIntel-Hub`.
5. **Agent safety / multi-agent / long-term memory** — reuses `agent-reliability-research`, mainly orchestration + evaluation.
6. **Remote sensing / interpretability / AIGC detection (SCI Q3–Q4)** — lowest bar, steadiest "safety-net publication," reuses `geospatial-fm-reliability-research`.

**Avoid**: all materials/chemistry/mechanical topics, real-robot execution, real-vehicle autonomous-driving closed-loop, training a video world model from scratch — these aren't money/time problems, they are things this machine structurally cannot do.

---

## 5. Three things to do before starting

1. **Fix torchvision** (required for any vision/diffusion work): reinstall a torchvision matching `torch 2.12.1+cu130`, verify that `python -c "import torchvision; torchvision.ops.nms"` doesn't error.
2. **Free up disk space**: only 579 GB left, HF/Ollama cache history takes up ~380 GB — clean it up before starting a new direction.
3. **Start with 1 Category-A idea + 1 Category-B idea** — don't spread out across many at once; a single 24 GB GPU can only seriously run one training job at a time.
