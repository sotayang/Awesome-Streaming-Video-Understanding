# Awesome-Streaming-Video-Understanding [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

## **Overview**

This repository provides a curated collection of research papers, models, and datasets focused on **Streaming (Online) Video Understanding**. The field aims to develop AI assistants capable of J.A.R.V.I.S.-like continuous multimodal perception and interaction. Unlike traditional offline video understanding, where models have access to the complete video beforehand, streaming models must operate under real-time, causal constraints: frames arrive sequentially, and decisions at any moment can only rely on past and present information, without the ability to rewind or preview future content.

This paradigm introduces two fundamental challenges:
1.  **Proactive Decision-Making (`When` to Act):** Determining the optimal moment to generate a response, ask for clarification, or remain silent.
2.  **Efficient Resource Management (`How` to Sustain):** Managing ever-growing context (memory/KV cache) and computational load for perpetual, real-time processing.

The repository is organized to reflect these core challenges and the supporting ecosystem:

*   **🔔 Proactive Streaming Models:** Approaches for deciding *when* to interact, including token-driven triggering (EOS), dedicated classifiers, perplexity validation, and visual-based detection.
*   **📺 Responsive Streaming Models:** Techniques for efficient *long-context* processing, covering KV cache management, hierarchical memory, retrieval-augmentation, and computational optimizations.
*   **📊 Benchmarks & Datasets:** Key datasets for evaluating capabilities in multi-turn dialogue, real-time captioning, and proactive timing.

This list serves as a reference for researchers and practitioners exploring the frontier of always-on, interactive video AI systems. Love this awesome list? Help others discover it by starring the repository! ⭐

### <font size=5><center><b> Table of Contents </b> </center></font>
- [🔔 Proactive Streaming Models](#-proactive-streaming-models)
  - [Token-Driven Triggering via EOS](#token-driven-triggering-via-eos--action-token)
  - [Dedicated Classification Heads / Detectors](#dedicated-classification-heads--detectors)
  - [Uncertainty \& Perplexity Validation](#uncertainty--perplexity-validation)
  - [Visual Change / Event-based Trigger](#visual-change--event-based-trigger)
- [📺 Responsive Streaming Models](#-responsive-streaming-models)
  - [KV Cache Management \& Eviction](#kv-cache-management--eviction)
  - [Hierarchical Memory \& Summarization](#hierarchical-memory--summarization)
  - [Retrieval-Augmented Mechanisms](#retrieval-augmented-mechanisms)
  - [Computational Efficiency \& Sparse Computing](#computational-efficiency--sparse-computing)
- [📊 Benchmarks \& Datasets](#-benchmarks--datasets)
  - [Multi-Turn Dialogue \& QA](#multi-turn-dialogue--qa)
  - [Real-time Captioning \& Narration](#real-time-captioning--narration)
  - [Proactive Response \& Timing Evaluation](#proactive-response--timing-evaluation)
- [📋 Complete Model List by Release Date](#complete-model-list-by-release-date)
- [📋 Complete Dataset List by Release Date](#complete-dataset-list-by-release-date)

---

## 🔔 Proactive Streaming Models

### Token-Driven Triggering via EOS / Action Token
*Models that decide actions (Speak, Wait, or others) by generating specific tokens or action probabilities within the sequence.* Typically, they learn through autoregressive prediction where an EOS token represents silence, while regular language tokens represent responses. This approach may potentially impact the model's general-purpose capabilities.

| Paper | Model | Date | Link | Venue | Method / Key Contribution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [Eyes Wide Open: Ego Proactive Video-LLM for Streaming Video](https://arxiv.org/abs/2510.14560) | VideoLLM-EyeWO | 2025/10 | [Link](https://zhangyl4.github.io/publications/eyes-wide-open/) | NeurIPS 2025 |  **Active Perception & Action**: Predicts 3 actions (Silence, Respond, **Ask-High-Res**); proactively requests high-res frames when uncertain to ensure just-in-time accuracy. |
| [Proactive Assistant Dialogue Generation from Streaming Egocentric Videos](https://arxiv.org/abs/2506.05904) | ProAssist | 2025/06 | [Link](https://github.com/pro-assist/ProAssist) | arXiv | **EOS-Based Trigger**: Predicts **[EOS]** token to remain silent or generates text to respond at each frame. Uses **Negative Frame Sub-sampling** to handle class imbalance between silence and speaking. |
| [LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale](https://arxiv.org/abs/2504.16030) | LiveCC | 2025/04 | [Link](https://showlab.github.io/livecc/) | CVPR 2025 | **EOS-Based**: Trains on large-scale streaming ASR data. At inference, the model predicts **[EOS]** to stay silent or generates commentary tokens frame-by-frame, enabling real-time play-by-play narration. |
| [AssistPDA: An Online Video Surveillance Assistant for Video Anomaly Prediction, Detection, and Analysis](https://arxiv.org/abs/2503.21904) | AssistPDA | 2025/03 | N/A | arXiv | **EOS-Based**: Predicts **[EOS]** probability to decide whether to output an anomaly alert/prediction. Features a **STRD** module to distill offline temporal reasoning into online inference. |
| [LION-FS: Fast & Slow Video-Language Thinker as Online Video Assistant](https://arxiv.org/abs/2503.03663) | LION-FS | 2025/03 | [Link](https://github.com/JiuTian-VL/LION-FS) | CVPR 2025 | **EOS-Based + Fast-Slow Architecture**: Uses a **Fast Path** to efficiently determine *when* to respond (via token prediction) and a **Slow Path** with multi-granularity keyframe augmentation to generate detailed responses only when needed. |
| [VideoLLM-MoD: Efficient Video-Language Streaming with Mixture-of-Depths Vision Computation](http://arxiv.org/abs/2408.16730) | VideoLLM-MoD     | 2024/08 | N/A                                                          | NeurIPS 2024 | **EOS-Based + MoD Efficiency**: Inherits **[EOS]** token prediction for proactive triggering. Key contribution is **Mixture-of-Depths**, dynamically skipping redundant vision token computation to enable efficient streaming. |
| [VideoLLM-online: Online Video Large Language Model for Streaming Video](https://arxiv.org/abs/2406.11816) | VideoLLM-online | 2024/06 | [Link](https://github.com/showlab/VideoLLM-online) | CVPR 2024 | **Streaming EOS**: Pioneered the **Streaming EOS** training objective. The model predicts an **[EOS]** token at each frame to decide whether to stay silent or generate a response, enabling real-time, proactive interaction. |



### Dedicated Classification Heads / Detectors
*Models that use a lightweight detector, router head, or auxiliary module to trigger responses.* A binary classification module determines whether to remain silent or to respond. 

| Paper | Model | Date | Link | Venue | Method / Key Contribution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [Open-ended Hierarchical Streaming Video Understanding with Vision Language Models](https://arxiv.org/abs/2509.12145) | OpenHOUSE | 2025/09 | N/A | ICCV 2025 | **Detector-Triggered Hierarchical Captioning**: Uses a lightweight **Streaming Module** (RNN) to detect action boundaries (hybrid actionness/progress). Triggers the frozen VLM only at detected boundaries to generate hierarchical (substep/step) descriptions. |
| [StreamAgent: Towards Anticipatory Agents for Streaming Video Understanding](https://arxiv.org/abs/2508.01875) | StreamAgent | 2025/08 | N/A | arXiv | **Agent-as-Detector**: Uses a separate, lightweight **Anticipatory Agent** (Small VLM) to act as a decision module. It plans and predicts future events to **trigger** the main responder only when necessary, decoupling decision from generation. |
| [StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Assistant](https://arxiv.org/abs/2505.05467) | StreamBridge | 2025/05 | [Link](https://github.com/apple/ml-streambridge) | NeurIPS 2025 | **Decoupled Activation Model**: Uses a separate, lightweight **Activation Model** (e.g., 0.5B LLaVA) to detect "when to speak" (triggering), allowing the main offline Video-LLM to be plug-and-play for proactive streaming. Also uses **Round-Decayed Compression** for memory. |
| [ViSpeak: Visual Instruction Feedback in Streaming Videos](https://arxiv.org/abs/2503.12769) | ViSpeak | 2025/03 | [Link](https://github.com/HumanMLLM/ViSpeak) | ICCV 2025 | **Classification Head Trigger**: Defines "Visual Instruction Feedback" tasks (e.g., visual wake-up, interruption). Uses a trained **binary classification head** (Informative Head) on top of the VLM to predict "when to speak" based on visual cues. |
| [StreamMind: Unlocking Full Frame Rate Streaming Video Dialogue through Event-Gated Cognition](https://arxiv.org/abs/2503.06220) | StreamMind | 2025/03 | [Link](https://aka.ms/StreamMind) | ICCV 2025 | **Cognition Gate**: Introduces an **Event-Gated** mechanism. A lightweight **Cognition Gate** (initialized from LLM shallow layers) continuously monitors the stream and only **triggers/invokes** the heavy LLM when relevant events occur, enabling 100 FPS processing. |
| [Dispider: Enabling Video LLMs with Active Real-Time Interaction via Disentangled Perception, Decision, and Reaction](https://arxiv.org/abs/2501.03218) | Dispider | 2025/01 | [Link](https://github.com/Mark12Ding/Dispider) | CVPR 2025 | **Disentangled Decision Module**: Decouples **Perception** (streaming), **Decision** (when to speak), and **Reaction** (generation) into asynchronous modules. Uses a lightweight decision model to trigger the heavy reaction model only when needed. |
| [ViSpeak: Visual Instruction Feedback in Streaming Videos](https://arxiv.org/abs/2503.12769) | ViSpeak | 2025/03 | [Link](https://github.com/HumanMLLM/ViSpeak) | ICCV 2025 | **Classification Head Trigger**: Defines "Visual Instruction Feedback" tasks (e.g., visual wake-up, interruption). Uses a trained **binary classification head** (Informative Head) on top of the VLM to predict "when to speak" based on visual cues. |
| [VideoLLM Knows When to Speak: Enhancing Time-Sensitive Video Comprehension with Video-Text Duet Interaction](https://arxiv.org/abs/2411.17991) | MMDuet | 2024/11 | [Link](https://github.com/yellow-binary-tree/MMDuet) | arXiv | **Dual-Head Trigger**: Trains two binary classification heads (**Informative Head** & **Relevance Head**) to decide when to interrupt the video stream and generate a response. Enables "Duet" interaction format. |
| [Streamlined Dense Video Captioning](https://arxiv.org/abs/1904.03870) | SDVC | 2019/04 | [Link](https://github.com/ranjaykrishna/densevid_eval) | CVPR 2019 | **Event Sequence Generation**: Uses an **Event Sequence Generation Network** (Pointer Net) to adaptively select a sequence of event proposals, which then triggers the captioning network. (Note: Offline method). |

### Uncertainty & Perplexity Validation
*Models that monitor PPL spikes or uncertainty scores to initiate interaction.* For previously spoken content, new frames are validated for perplexity: low perplexity indicates the content remains unchanged, thus no repeated decoding is needed (silent); high perplexity indicates new content in the frame, triggering a response.

| Paper | Model | Date | Link | Venue | Method / Key Contribution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [LiveStar: Live Streaming Assistant for Real-World Online Video Understanding](https://arxiv.org/abs/2511.05299) | LiveStar | 2025/11 | [Link](https://github.com/sotayang/LiveStar) | NeurIPS 2025 | **PPL-Based Verification (SVeD)**: Uses **Streaming Verification Decoding (SVeD)** which calculates the **perplexity (PPL)** of the generated caption to verify its validity. If PPL indicates high confidence/necessity, it triggers a response; otherwise, it stays silent. |

### Visual Change / Event-based Trigger
*Models that trigger responses based on significant changes in the visual stream or detected events.* Frames with substantial visual changes often trigger new responses, while frames with minimal changes typically correspond to unchanged content from before.

| Paper | Model | Date | Link | Venue | Method / Key Contribution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [TimeChat-Online: 80% Visual Tokens are Naturally Redundant in Streaming Videos](https://arxiv.org/abs/2504.17343) | TimeChat-Online | 2025/04 | [Link](https://timechat-online.github.io/) | ACM MM 2025 | **Visual Change Trigger**: Uses **Differential Token Drop (DTD)** to prune redundant tokens. Monitors the **token drop ratio**; sudden drops indicate scene transitions, which serve as natural **triggers** for proactive responding. |

---

## 📺 Responsive Streaming Models

### KV Cache Management & Eviction
*Methods focusing on optimizing the KV cache by evicting less important tokens (e.g., Heavy Hitter, Sliding Window).*

| Paper | Model | Date | Link | Venue | Method / Key Contribution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [StreamingVLM: Real-Time Understanding for Infinite Video Streams](https://arxiv.org/abs/2510.09608) | StreamingVLM | 2025/10 | [Link](https://github.com/mit-han-lab/streaming-vlm) | arXiv | **Streaming-Aware KV Cache**: Uses **Attention Sinks** + Sliding Window (Long Text + Short Vision) with **Contiguous RoPE** to enable infinite streaming without memory explosion or positional drift. Trains with overlapped-chunk full attention. |
| [StreamMem: Query-Agnostic KV Cache Memory for Streaming Video Understanding](https://arxiv.org/abs/2508.15717) | StreamMem | 2025/08 | [Link](https://yangyanl.ai/streammem/) | arXiv | **Query-Agnostic Compression**: Uses standard chat template tokens as **Proxy Queries** to calculate attention scores for **Pruning** and **Merging** KV cache, maintaining a fixed memory budget without needing the actual user query. |
| [StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling](https://arxiv.org/abs/2507.05240) | StreamVLN | 2025/07 | [Link](https://streamvln.github.io/) | arXiv | **SlowFast Context (Pruning)**: Combines a **Sliding Window** (Fast Path) for recent dialogue with a **3D-Aware Token Pruning** (Slow Path) to compress historical visual states into a compact memory, enabling long-horizon navigation. |
| [InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding](https://arxiv.org/abs/2506.15745) | InfiniPot-V | 2025/06 | [Link](https://github.com/aiha-lab/InfiniPot-V) | NeurIPS 2025 | **Continual KV Compression**: Maintains a fixed memory budget by periodically compressing the KV cache using **Temporal-axis Redundancy (TaR)** (evicting repetitive frames) and **Value-Norm (VaN)** (keeping semantically important tokens). |


### Hierarchical Memory & Summarization
*Methods that compress history into events, super-tokens, or hierarchical structures.*

| Paper | Model | Date | Link | Venue | Method / Key Contribution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [video-SALMONN S: Streaming Audio-Visual LLMs Beyond Length Limits via Memory](https://arxiv.org/abs/2510.11129) | video-SALMONN S | 2025/10 | N/A | arXiv | **TTT Memory**: Uses **Test-Time Training (TTT)** layers to compress video history into model weights (hidden state) + **Prompt-dependent memory reading** to extract relevant info from fixed-size memory. First to process >3h video at 1FPS. |
| [StreamForest: Efficient Online Video Understanding with Persistent Event Memory](https://arxiv.org/abs/2509.24871) | StreamForest | 2025/09 | [Link](https://github.com/MCG-NJU/StreamForest) | NeurIPS 2025 | **Tree-Structured Event Memory**: Organizes video frames into a **Persistent Event Memory Forest** (tree structure). Adaptively **merges** event nodes based on penalty functions (time, similarity, merge count) to maintain long-term history within a fixed token budget. |
| [Open-ended Hierarchical Streaming Video Understanding with Vision Language Models](https://arxiv.org/abs/2509.12145) | OpenHOUSE | 2025/09 | N/A | ICCV 2025 | **Hierarchical Understanding**: Decouples streaming into a lightweight perception module (detects action boundaries) and a VLM. Uses **Context Memory** to store and **retrieve** hierarchical history (substeps/steps) to generate coarse-to-fine descriptions. |
| [OVG-HQ: Online Video Grounding with Hybrid-modal Queries](https://arxiv.org/abs/2508.11903) | OVG-HQ-Unify | 2025/08 | [Link](https://github.com/maojiaqi2324/OVG-HQ) | ICCV 2025 | **Parametric Memory (TTT)**: Uses a **Parametric Memory Block (PMB)** instantiated with a **Test-Time Training (TTT)** layer to compress historical video context into network parameters for online grounding. Supports hybrid-modal queries (text/image/video). |
| [Flash-VStream: Efficient Real-Time Understanding for Long Video Streams](https://arxiv.org/abs/2506.23825) | Flash-VStream | 2025/06 | [Link](https://github.com/IVGSZ/Flash-VStream) | ICCV 2025 | **Flash Memory**: Two-process framework with 1. **Context Synopsis Memory (CSM)**: Compresses history via K-means clustering (summarization). 2. **Detail Augmentation Memory (DAM)**: **Retrieves** high-res spatial details for key frames based on CSM distribution. |
| [Memory-efficient Streaming VideoLLMs for Real-time Procedural Video Understanding](https://arxiv.org/abs/2504.13915) | ProVideLLM | 2025/04 | [Link](https://dibschat.github.io/ProVideLLM/) | ICCV 2025 | **Verbalized Memory**: Maintains a multimodal cache by **verbalizing** long-term video history into text steps (summarization) while keeping short-term history as visual tokens (extracted by DETR-QFormer), enabling extremely efficient streaming. |
| [Streaming Video Understanding and Multi-round Interaction with Memory-enhanced Knowledge](https://arxiv.org/abs/2501.13468) | StreamChat | 2025/01 | [Link](https://github.com/hmxiong/StreamChat) | ICLR 2025 | **Hierarchical Memory Tree**: Builds a **long-term memory tree** by clustering and captioning video chunks. Uses a parallel scheduling system to update memory and **retrieve** relevant context for multi-turn dialogue. |
| [Online Video Understanding: OVBench and VideoChat-Online](https://arxiv.org/abs/2501.00584) | VideoChat-Online | 2025/01 | [Link](https://videochat-online.github.io/) | CVPR 2025 | **Pyramid Memory Bank**: Uses a hierarchical memory ($m_t, m_{main}, m_s$) with progressive abstraction (pooling resolution/rate) to balance spatial and temporal details. Employs **Frame Eviction & Down Writing** to compress older frames into lower-resolution layers. |
| [VideoLLaMB: Long Streaming Video Understanding with Recurrent Memory Bridges](https://arxiv.org/abs/2409.01071) | VideoLLaMB | 2024/09 | [Link](https://github.com/bigai-nlco/VideoLLaMB) | ICCV 2025 | **Recurrent Memory Bridge**: Uses **SceneTiling** to segment video into semantic clips. Compresses clips into **Memory Tokens** via recurrent bridge layers, which are periodically updated via retrieval, enabling long-context understanding with linear memory scaling. |
| [Streaming Long Video Understanding with Large Language Models](https://arxiv.org/abs/2405.16009) | VideoStreaming | 2024/05 | N/A | NeurIPS 2024 | **Memory-Propagated Encoding**: Segments video into clips and encodes them into **condensed memories** using a small LLM, with memory propagated recursively. Uses **Adaptive Memory Selection** to retrieve relevant clips for QA. |
| [Synchronized Video Storytelling: Generating Video Narrations with Structured Storyline](https://arxiv.org/abs/2405.14040) | VideoNarrator | 2024/05 | [Link](https://github.com/alibaba/alimama-video-narrator) | ACL 2024 | **Memory Consolidation**: Defines "Synchronized Video Storytelling". Uses **Memory Consolidation** to merge past visual tokens into fixed-length memory, and generates narrations guided by a **structured storyline**. |
| [Streaming Dense Video Captioning](https://arxiv.org/abs/2404.01297) | StreamingDVC | 2024/04 | [Link](https://github.com/google-research/scenic/tree/main/scenic/projects/streaming_dvc) | CVPR 2024 | **Clustering-Based Memory**: Compresses incoming visual tokens into a fixed-size memory using **K-means clustering**. Uses a streaming decoding algorithm to output captions before the entire video is processed. |


### Retrieval-Augmented Mechanisms
*Methods employing external memory banks and retrieval systems.*

| Paper | Model | Date | Link | Venue | Method / Key Contribution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [CacheFlow: Compressive Streaming Memory for Efficient Long-Form Video Understanding](https://arxiv.org/abs/2511.13644) | CacheFlow | 2025/11 | N/A | arXiv | **Consensus-First Retrieval**: Offloads KV cache to CPU. Compresses old blocks using a **GRU-based** memory. Retrieves top-K blocks based on a **consensus** score from shallow and deep layers, rehydrating them to GPU for inference. |
| [StreamKV: Streaming Video Question-Answering with Segment-based KV Cache Retrieval and Compression](https://arxiv.org/abs/2511.07278) | StreamKV | 2025/11 | [Link](https://github.com/sou1p0wer/StreamKV) | arXiv | **Segment-based Retrieval**: Partitions video into semantic segments and uses a **Guidance Prompt** to compress KV cache. Stores compressed KVs in a bank and **retrieves** relevant segments based on user query for QA. |
| [StreamingTOM: Streaming Token Compression for Efficient Video Understanding](https://arxiv.org/abs/2510.18269) | StreamingTOM | 2025/10 | [Link](https://yige24.github.io/StreamingTOM) | arXiv | **Two-stage Framework**: 1. **CTR (Pre-LLM)**: Prunes input tokens based on temporal redundancy to speed up prefill. 2. **OQM (Post-LLM)**: Stores 4-bit quantized KV groups and **retrieves** Top-K relevant groups on-demand for decoding. |
| [Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs](https://arxiv.org/abs/2510.17364) | rLiVS | 2025/10 | N/A | arXiv | **Caption-Based Retrieval**: 1. **Token Selection**: Uses LLM attention scores to select top ~5% visual tokens and passes them recurrently. 2. **Retrieval**: Generates captions for clips and **retrieves** top-K text captions to answer user queries, avoiding heavy KV storage. |
| [CogStream: Context-guided Streaming Video Question Answering](https://arxiv.org/abs/2506.10516) | CogReasoner | 2025/06 | [Link](https://github.com/LiamZhao326/CogStream) | arXiv | **Dialogue Retrieval & Visual Compression**: 1. **Visual Stream Compression**: Clusters frames into events and compresses based on question relevance. 2. **Historic Dialogue Retrieval**: Uses LLM to **retrieve** relevant past QA pairs to support current reasoning. |
| [LiveVLM: Efficient Online Video Understanding via Streaming-Oriented KV Cache and Retrieval](https://arxiv.org/abs/2505.15269) | LiveVLM | 2025/05 | N/A | arXiv | **Streaming-Oriented KV Cache & Retrieval**: 1. Compresses video KV pairs via attention-based pruning and frame-wise merging. 2. **Retrieves** relevant long-term KV chunks based on query attention scores to answer questions efficiently. |
| [Streaming Video Question-Answering with In-context Video KV-Cache Retrieval](https://arxiv.org/abs/2503.00540) | ReKV | 2025/03 | [Link](https://github.com/Becomebright/ReKV) | ICLR 2025 | **KV-Cache Retrieval**: Offloads video KV caches to CPU/Disk. Upon receiving a query, it **retrieves** and reloads only the relevant KV caches to GPU for efficient answer generation, decoupling encoding from QA. |






### Computational Efficiency & Sparse Computing
*Methods reducing FLOPs via dynamic compute, sparse attention, or efficient backbone designs.*

| Paper | Model | Date | Link | Venue | Method / Key Contribution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [Learning Streaming Video Representation via Multitask Training](https://arxiv.org/abs/2504.20041) | StreamFormer | 2025/04 | [Link](https://github.com/Go2Heart/StreamFormer) | ICCV 2025 | **Efficient Streaming Backbone**: Introduces **Causal Temporal Attention** into Vision Transformers to enable efficient frame-by-frame processing. Trained via **Multitask Learning** (classification, detection, segmentation) to learn robust spatiotemporal representations. |
| [StreamChat: Chatting with Streaming Video](https://arxiv.org/abs/2412.08646) | StreamChat | 2024/12 | [Link](https://jihaonew.github.io/projects/streamchat.html) | CVPR 2024  | **Cross-Attention Streaming Architecture**: Dynamically updates visual context during decoding via lightweight cross-attention, enhanced with **V-FFN refinement** and **parallel 3D-RoPE** for stable temporal alignment, enabling real-time streaming interaction without trigger modules. |

---

## 📊 Benchmarks & Datasets


### Multi-Turn Dialogue & QA

| Paper | Dataset | Date | Link | Venue | Tasks |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [StreamingCoT: A Dataset for Temporal Dynamics and Multimodal Chain-of-Thought Reasoning in Streaming VideoQA](https://arxiv.org/abs/2510.25332) | StreamingCoT | 2025/10 | [Link](https://github.com/Fleeting-hyh/StreamingCoT) | ACM MM 2025 | Streaming VideoQA, CoT Reasoning | Streaming VideoQA, CoT Reasoning |
| [StreamForest: Efficient Online Video Understanding with Persistent Event Memory](https://arxiv.org/abs/2509.24871) | ODV-Bench | 2025/09 | [Link](https://huggingface.co/datasets/MCG-NJU/ODV-Bench) | NeurIPS 2025 | Streaming VideoQA (Autonomous Driving), Real-time Perception, Future Prediction (Risk/Trajectory), Past Memory |
| [OST-Bench: Evaluating the Capabilities of MLLMs in Online Spatio-temporal Scene Understanding](https://arxiv.org/abs/2507.07984) | OST-Bench | 2025/10 | [Link](https://rbler1234.github.io/OSTBench.github.io/) | NeurIPS 2025 | Online Spatio-Temporal QA, Agent State Estimation, 3D Spatial Reasoning, Memory Retrieval |
| [RTV-Bench: Evaluating the Capabilities of MLLMs in Online Spatio-temporal Scene Understanding](https://arxiv.org/abs/2505.02064) | RTV-Bench | 2025/05 | [Link](https://github.com/LJungang/RTV-Bench) | NeurIPS 2025 | Real-Time Video Reasoning, Sport / Driving / Ego Scenario, hierarchical Evaluation |
| [Online Video Understanding: OVBench and VideoChat-Online](https://arxiv.org/abs/2501.00584) | OVBench | 2025/04 | [Link](https://videochat-online.github.io/) | CVPR 2025 | Online VideoQA, Past Memory, Future Prediction, Spatial Perception |
| [SVBench: A Benchmark with Temporal Multi-Turn Dialogues for Streaming Video Understanding](https://arxiv.org/abs/2502.10810) | SVBench | 2025/02 | [Link](https://sotayang.github.io/SVBench/) | ICLR 2025 | Streaming VideoQA, Temporal Multi-Turn Dialogue, Long-Context Reasoning |
| [STREAMCHAT: Streaming Video Understanding and Multi-Round Interaction with Memory-Enhanced Knowledge](https://arxiv.org/abs/2501.13468) | STREAMBENCH | 2025/01 | [Link](https://github.com/Panda-Peter/StreamChat) | ICLR 2025 | Streaming VideoQA, Multi-turn Dialogue, Long/Short-term Memory, Object Search |
| [StreamingBench: Assessing the Gap for MLLMs to Achieve Streaming Video Understanding](https://arxiv.org/abs/2411.03628) | StreamingBench | 2024/11 | [Link](https://github.com/THUNLP-MT/StreamingBench) | arXiv | Real-time Visual QA, Omni-source QA, Contextual QA |


### Real-time Captioning & Narration

| Paper | Dataset | Date | Link | Venue | Tasks |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [LiveStar: Live Streaming Assistant for Real-World Online Video Understanding](https://arxiv.org/abs/2511.05299) | OmniStar-RNG | 2025/11 | [Link](https://huggingface.co/datasets/yzy666/OmniStar-RNG) | NeurIPS 2025 | Real-time Narration, Streaming Dense Captioning, Streaming Video-Text Alignment |
| [LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale](https://arxiv.org/abs/2504.16030) | Live-CC-5M | 2025/04 | [Link](https://huggingface.co/datasets/chenjoya/Live-CC-5M) | CVPR 2025 | Large-scale Pre-training, Streaming Captioning (ASR-based), Video-Text Alignment |
| [LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale](https://arxiv.org/abs/2504.16030) | Live-WhisperX-526K | 2025/04 | [Link](https://huggingface.co/datasets/chenjoya/Live-WhisperX-526K) | CVPR 2025 | Real-time Video Commentary, Instruction Tuning, Dense Streaming Captioning |


### Proactive Response & Timing Evaluation

| Paper | Dataset | Date | Link | Venue | Tasks |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](https://arxiv.org/abs/2512.01707) | StreamGaze | 2025/12 | N/A | arXiv | Gaze-Triggered Alert, Object Transition Prediction, Gaze Sequence Matching |
| [Eyes Wide Open: Ego Proactive Video-LLM for Streaming Video](https://arxiv.org/abs/2510.14560) | ESTP-Bench | 2025/10 | [Link](https://zhangyl4.github.io/publications/eyes-wide-open/) | NeurIPS 2025 | Proactive QA, Just-in-Time Response, Egocentric Reasoning |
| [ProactiveVideoQA: A Comprehensive Benchmark Evaluating Proactive Interactions in Video Large Language Models](https://arxiv.org/abs/2507.09313) | ProactiveVideoQA | 2025/07 | [Link](https://github.com/yellow-binary-tree/ProactiveVideoQA) | arXiv | Proactive VideoQA (Web/Ego/TV), Response Timing Evaluation, Anomaly Detection |
| [Proactive Assistant Dialogue Generation from Streaming Egocentric Videos](https://arxiv.org/abs/2506.05904) | PROASSIST | 2025/06 | [Link](https://pro-assist.github.io/) | arXiv | Proactive Task Guidance, Streaming Dialogue, Response Timing (When to speak) |
| [RTV-Bench: Evaluating the Capabilities of MLLMs in Online Spatio-temporal Scene Understanding](https://arxiv.org/abs/2505.02064) | RTV-Bench | 2025/05 | [Link](https://github.com/LJungang/RTV-Bench) | NeurIPS 2025 | Real-Time Video Reasoning, Sport / Driving / Ego Scenario, hierarchical Evaluation |
| [OmniMMI: A Comprehensive Multi-modal Interaction Benchmark in Streaming Video Contexts](https://arxiv.org/abs/2503.22952) | OmniMMI | 2025/03 | [Link](https://omnimmi.github.io/) | arXiv | Streaming Video Understanding (State Grounding, Action Planning), Proactive Reasoning (Alerting, Turn-Taking) |
| [AssistPDA: An Online Video Surveillance Assistant for Video Anomaly Prediction, Detection, and Analysis](https://arxiv.org/abs/2503.21904) | VAPDA-127K | 2025/03 | N/A  | arXiv | Proactive Anomaly Prediction, Online Anomaly Detection, Interactive Anomaly Analysis |
| [OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?](https://arxiv.org/abs/2501.05510) | OVO-Bench | 2025/01 | [Link](https://github.com/JoeLeelyf/OVO-Bench) | CVPR 2025 | Forward Active Responding (When to Answer), Backward Tracing, Real-time Perception |
| [VideoLLM Knows When to Speak: Enhancing Time-Sensitive Video Comprehension with Video-Text Duet Interaction Format](https://arxiv.org/abs/2411.17991) | MMDuetIT / MAGQA | 2024/11 | [Link](https://huggingface.co/datasets/wangyueqian/MMDuetIT) | arXiv | Multi-Answer Grounded QA, Proactive Response Generation |

### Complete Model List by Release Date

<details> <summary>Models</summary>

| Paper                                                        | Model            | Date    | Link                                                         | Venue        |
| ------------------------------------------------------------ | ---------------- | ------- | ------------------------------------------------------------ | ------------ |
| [LiveStar: Live Streaming Assistant for Real-World Online Video Understanding](https://arxiv.org/abs/2511.05299) | LiveStar         | 2025/11 | [Link](https://github.com/sotayang/LiveStar)                 | NeurIPS 2025 |
| [CacheFlow: Compressive Streaming Memory for Efficient Long-Form Video Understanding](https://arxiv.org/abs/2511.13644) | CacheFlow        | 2025/11 | N/A                                                          | arXiv        |
| [StreamKV: Streaming Video Question-Answering with Segment-based KV Cache Retrieval and Compression](https://arxiv.org/abs/2511.07278) | StreamKV         | 2025/11 | [Link](https://github.com/sou1p0wer/StreamKV)                | AAAI 2026    |
| [Eyes Wide Open: Ego Proactive Video-LLM for Streaming Video](https://arxiv.org/abs/2510.14560) | VideoLLM-EyeWO   | 2025/10 | [Link](https://zhangyl4.github.io/publications/eyes-wide-open/) | NeurIPS 2025 |
| [StreamingVLM: Real-Time Understanding for Infinite Video Streams](https://arxiv.org/abs/2510.09608) | StreamingVLM     | 2025/10 | [Link](https://github.com/mit-han-lab/streaming-vlm)         | arXiv        |
| [StreamingTOM: Streaming Token Compression for Efficient Video Understanding](https://arxiv.org/abs/2510.18269) | StreamingTOM     | 2025/10 | [Link](https://yige24.github.io/StreamingTOM/)               | arXiv        |
| [Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs](https://arxiv.org/abs/2510.17364) | rLiVS            | 2025/10 | N/A                                                          | arXiv        |
| [video-SALMONN S: Streaming Audio-Visual LLMs Beyond Length Limits via Memory](https://arxiv.org/abs/2510.11129) | video-SALMONN S  | 2025/10 | N/A                                                          | arXiv        |
| [StreamForest: Efficient Online Video Understanding with Persistent Event Memory](https://arxiv.org/pdf/2509.24871) | StreamForest     | 2025/09 | [Link](https://github.com/MCG-NJU/StreamForest)              | NeurIPS 2025 |
| [Open-ended Hierarchical Streaming Video Understanding with Vision Language Models](https://arxiv.org/abs/2509.12145) | OpenHOUSE        | 2025/09 | N/A                                                          | ICCV 2025    |
| [StreamMem: Query-Agnostic KV Cache Memory for Streaming Video Understanding](http://arxiv.org/abs/2508.15717) | StreamMem        | 2025/08 | [Link](https://yangyanl.ai/streammem/)                       | arXiv        |
| [StreamAgent: Towards Anticipatory Agents for Streaming Video Understanding](http://arxiv.org/abs/2508.01875) | StreamAgent      | 2025/08 | N/A                                                          | arXiv        |
| [OVG-HQ: Online Video Grounding with Hybrid-modal Queries](https://arxiv.org/abs/2508.11903) | OVG-HQ-Unify     | 2025/08 | [Link](https://github.com/maojiaqi2324/OVG-HQ)               | ICCV 2025    |
| [StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling](https://arxiv.org/abs/2507.05240) | StreamVLN        | 2025/07 | [Link](https://streamvln.github.io/)                         | arXiv        |
| [InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding](http://arxiv.org/abs/2506.15745) | InfiniPot-V      | 2025/06 | [Link](https://github.com/aiha-lab/InfiniPot-V)              | NeurIPS 2025 |
| [CogStream: Context-guided Streaming Video Question Answering](http://arxiv.org/abs/2506.10516) | CogStream        | 2025/06 | [Link](https://github.com/LiamZhao326/CogStream)             | arXiv        |
| [Proactive Assistant Dialogue Generation from Streaming Egocentric Videos](http://arxiv.org/abs/2506.05904) | N/A              | 2025/06 | [Link](https://github.com/pro-assist/ProAssist)              | arXiv        |
| [Flash-VStream: Efficient Real-Time Understanding for Long Video Streams](http://arxiv.org/abs/2506.23825) | Flash-VStream    | 2025/06 | [Link](https://github.com/IVGSZ/Flash-VStream)               | ICCV 2025    |
| [StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Assistant](https://arxiv.org/abs/2505.05467) | StreamBridge     | 2025/05 | [Link](https://github.com/apple/ml-streambridge)             | NeurIPS 2025 |
| [LiveVLM: Efficient Online Video Understanding via Streaming-Oriented KV Cache and Retrieval](https://arxiv.org/pdf/2505.15269) | LiveVLM          | 2025/05 | N/A                                                          | arXiv        |
| [TimeChat-Online: 80% Visual Tokens are Naturally Redundant in Streaming Videos](https://arxiv.org/abs/2504.17343) | TimeChat-Online  | 2025/04 | [Link](https://timechat-online.github.io/)                   | ACM MM 2025  |
| [Learning Streaming Video Representation via Multitask Training](http://arxiv.org/abs/2504.20041) | Streamformer     | 2025/04 | [Link](https://github.com/Go2Heart/StreamFormer)             | ICCV 2025    |
| [LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale](http://arxiv.org/abs/2504.16030) | LiveCC           | 2025/04 | [Link](https://showlab.github.io/livecc/)                    | CVPR 2025    |
| [Memory-efficient Streaming VideoLLMs for Real-time Procedural Video Understanding](https://arxiv.org/abs/2504.13915) | ProVideLLM       | 2025/04 | [Link](https://dibschat.github.io/ProVideLLM/)               | ICCV 2025    |
| [ViSpeak: Visual Instruction Feedback in Streaming Videos](http://arxiv.org/abs/2503.12769) | ViSpeak          | 2025/03 | [Link](https://github.com/HumanMLLM/ViSpeak)                 | ICCV 2025    |
| [AssistPDA: An Online Video Surveillance Assistant for Video Anomaly Prediction, Detection, and Analysis](https://arxiv.org/abs/2503.21904) | AssistPDA        | 2025/03 | N/A                                                          | arXiv        |
| [VideoScan: Enabling Efficient Streaming Video Understanding via Frame-level Semantic Carriers](https://arxiv.org/abs/2503.09387) | VideoScan        | 2025/03 | [Link](https://github.com/LyliAgave/VideoScan)               | arXiv        |
| [LION-FS: Fast-Slow Video Language Thinker as Online Video Assistant](http://arxiv.org/abs/2503.03663) | LION-FS          | 2025/03 | [Link](https://github.com/JiuTian-VL/LION-FS)                | CVPR 2025    |
| [StreamMind: Unlocking Full Frame Rate Streaming Video Dialogue through Event-Gated Cognition](http://arxiv.org/abs/2503.06220) | StreamMind       | 2025/03 | [Link](https://aka.ms/StreamMind)                            | ICCV 2025    |
| [Streaming Video Question-Answering with In-context Video KV-Cache Retrieval](https://arxiv.org/abs/2503.00540) | ReKV             | 2025/03 | [Link](https://github.com/Becomebright/ReKV)                 | ICLR 2025    |
| [SVBench: A Benchmark with Temporal Multi-Turn Dialogues for Streaming Video Understanding](https://arxiv.org/abs/2502.10810) | StreamingChat    | 2025/02 | [Link](https://github.com/sotayang/SVBench)                  | ICLR 2025    |
| [Streaming Video Understanding and Multi-round Interaction with Memory-enhanced Knowledge](http://arxiv.org/abs/2501.13468) | StreamChat       | 2025/01 | [Link](https://github.com/hmxiong/StreamChat)                | ICLR 2025    |
| [Dispider: Enabling Video LLMs with Active Real-Time Interaction via Disentangled Perception, Decision, and Reaction](http://arxiv.org/abs/2501.03218) | Dispider         | 2025/01 | [Link](https://github.com/Mark12Ding/Dispider)               | CVPR 2025    |
| [Online Video Understanding: OVBench and VideoChat-Online](https://arxiv.org/abs/2501.00584) | VideoChat-Online | 2025/01 | [Link](https://videochat-online.github.io/)                  | CVPR 2025    |
| [StreamChat: Chatting with Streaming Video](https://arxiv.org/abs/2412.08646) | StreamChat       | 2024/12 | [Link](https://jihaonew.github.io/projects/streamchat.html)  | CVPR 2024    |
| [VideoLLM Knows When to Speak: Enhancing Time-Sensitive Video Comprehension with Video-Text Duet Interaction](http://arxiv.org/abs/2411.17991) | MMDuet           | 2024/11 | [Link](https://github.com/yellow-binary-tree/MMDuet)         | arXiv        |
| [VideoLLaMB: Long Streaming Video Understanding with Recurrent Memory Bridges](http://arxiv.org/abs/2409.01071) | VideoLLaMB       | 2024/09 | [Link](https://github.com/bigai-nlco/VideoLLaMB)             | ICCV 2025    |
| [VideoLLM-MoD: Efficient Video-Language Streaming with Mixture-of-Depths Vision Computation](http://arxiv.org/abs/2408.16730) | VideoLLM-MoD     | 2024/08 | N/A                                                          | NeurIPS 2024 |
| [VideoLLM-online: Online Video Large Language Model for Streaming Video](https://arxiv.org/abs/2406.11816) | VideoLLM-online  | 2024/06 | [Link](https://github.com/showlab/VideoLLM-online)           | CVPR 2024    |
| [Streaming Long Video Understanding](https://arxiv.org/pdf/2405.16009) | VideoStreaming   | 2024/05 | N/A                                                          | NeurIPS 2024 |
| [Synchronized Video Storytelling: Generating Video Narrations with Structured Storyline](https://arxiv.org/abs/2405.14040) | VideoNarrator    | 2024/05 | [Link](https://github.com/alibaba/alimama-video-narrator)    | ACL 2024     |
| [Streaming Dense Video Captioning](https://arxiv.org/abs/2404.01297) | N/A              | 2024/04 | [Link](https://github.com/google-research/scenic/tree/main/scenic/projects/streaming_dvc) | CVPR 2024    |
| [Streamlined Dense Video Captioning](https://arxiv.org/abs/1904.03870) | SDVC             | 2019/04 | [Link](https://github.com/ranjaykrishna/densevid_eval)       | CVPR 2019    |

</details>

### Complete Dataset List by Release Date
<details> <summary>Benchmarks & Datasets</summary>

| Paper                                                        | Dataset            | Date    | Link                                                         | Venue        |
| ------------------------------------------------------------ | ------------------ | ------- | ------------------------------------------------------------ | ------------ |
| [StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](https://arxiv.org/abs/2512.01707) | StreamGaze         | 2025/12 | N/A                                                          | arXiv        |
| [LiveStar: Live Streaming Assistant for Real-World Online Video Understanding](https://arxiv.org/abs/2511.05299) | OmniStar-RNG       | 2025/11 | [Link](https://huggingface.co/datasets/yzy666/OmniStar-RNG)  | NeurIPS 2025 |
| [StreamingCoT: A Dataset for Temporal Dynamics and Multimodal Chain-of-Thought Reasoning in Streaming VideoQA](https://arxiv.org/abs/2510.25332) | StreamingCoT       | 2025/10 | [Link](https://github.com/Fleeting-hyh/StreamingCoT)         | ACM MM 2025  |
| [Eyes Wide Open: Ego Proactive Video-LLM for Streaming Video](https://arxiv.org/abs/2510.14560) | ESTP‑Bench         | 2025/10 | [Link](https://zhangyl4.github.io/publications/eyes-wide-open/) | NeurIPS 2025 |
| [StreamForest: Efficient Online Video Understanding with Persistent Event Memory](https://arxiv.org/pdf/2509.24871) | ODV-Bench          | 2025/09 | [Link](https://huggingface.co/datasets/MCG-NJU/ODV-Bench)    | NeurIPS 2025 |
| [OST-Bench: Evaluating the Capabilities of MLLMs in Online Spatio-temporal Scene Understanding](https://arxiv.org/pdf/2507.07984) | OST-Bench          | 2025/07 | [Link](https://rbler1234.github.io/OSTBench.github.io/)      | NeurIPS 2025 |
| [ProactiveVideoQA: A Comprehensive Benchmark Evaluating Proactive Interactions in Video Large Language Models](https://arxiv.org/abs/2507.09313) | ProactiveVideoQA   | 2025/07 | [Link](https://github.com/yellow-binary-tree/ProactiveVideoQA) | arXiv        |
| [Proactive Assistant Dialogue Generation from Streaming Egocentric Videos](http://arxiv.org/abs/2506.05904) | ProAssist          | 2025/06 | [Link](https://github.com/pro-assist/ProAssist)              | arXiv        |
| [RTV-Bench: Evaluating the Capabilities of MLLMs in Online Spatio-temporal Scene Understanding](https://arxiv.org/abs/2505.02064) | RTV-Bench | 2025/05 | [Link](https://github.com/LJungang/RTV-Bench) | NeurIPS 2025 | Real-Time Video Reasoning, Sport / Driving / Ego Scenario, hierarchical Evaluation |
| [LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale](http://arxiv.org/abs/2504.16030) | Live-WhisperX-526K | 2025/04 | [Link](https://huggingface.co/datasets/chenjoya/Live-WhisperX-526K) | CVPR 2025    |
| [LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale](http://arxiv.org/abs/2504.16030) | Live-CC-5M         | 2025/04 | [Link](https://huggingface.co/datasets/chenjoya/Live-CC-5M)  | CVPR 2025    |
| [OmniMMI: A Comprehensive Multi-modal Interaction Benchmark in Streaming Video Contexts](https://arxiv.org/abs/2503.22952) | OmniMMI            | 2025/03 | [Link](https://omnimmi.github.io/)                           | CVPR 2025    |
| [AssistPDA: An Online Video Surveillance Assistant for Video Anomaly Prediction, Detection, and Analysis](https://arxiv.org/abs/2503.21904) | VAPDA-127K         | 2025/03 | N/A                                                          | arXiv        |
| [SVBench: A Benchmark with Temporal Multi-Turn Dialogues for Streaming Video Understanding](https://arxiv.org/abs/2502.10810) | SVBench            | 2025/02 | [Link](https://sotayang.github.io/SVBench/)                  | ICLR 2025    |
| [Online Video Understanding: OVBench and VideoChat-Online](https://arxiv.org/abs/2501.00584) | OVBench            | 2025/01 | [Link](https://videochat-online.github.io/)                  | CVPR 2025    |
| [OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?](https://arxiv.org/abs/2501.05510) | OVO-Bench          | 2025/01 | [Link](https://github.com/JoeLeelyf/OVO-Bench)               | CVPR 2025    |
| [Streaming Video Understanding and Multi-round Interaction with Memory-enhanced Knowledge](http://arxiv.org/abs/2501.13468) | StreamBench        | 2025/01 | [Link](https://github.com/hmxiong/StreamChat)                | ICLR 2025    |
| [StreamingBench: Assessing the Gap for MLLMs to Achieve Streaming Video Understanding](https://arxiv.org/abs/2411.03628) | StreamingBench     | 2024/11 | [Link](https://github.com/THUNLP-MT/StreamingBench)          | arXiv        |
| [VideoLLM Knows When to Speak: Enhancing Time-Sensitive Video Comprehension with Video-Text Duet Interaction](http://arxiv.org/abs/2411.17991) | MMDuetIT           | 2024/11 | [Link](https://huggingface.co/datasets/wangyueqian/MMDuetIT) | arXiv        |

</details>

## 🚀 Contributing  
We welcome contributions! To add a resource, you can:  

- Open a pull request with a clear title and brief description of your changes.  
- Open an issue with a clear title and short explanation.  

If you notice any errors, feel free to open an issue — we apologize in advance for any inconvenience.  

## ❤️ Contact  
If you have suggestions or find this project useful, we’d love to hear from you.  
Email: yangzhenyu2022@ia.ac.cn and zhangkr2025@shanghaitech.edu.cn


