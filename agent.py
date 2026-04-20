# agent.py

import os
from typing import TypedDict, List
from dotenv import load_dotenv
load_dotenv()

# LLM
print("LOADING AGENT FILE...")
from langchain_groq import ChatGroq

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Embeddings + DB
from sentence_transformers import SentenceTransformer
import chromadb

# Messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key="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"
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="knowledge_base")

# ADD YOUR DOCUMENTS HERE (copy from notebook)
raw_docs = [
    {
        "id": "doc_001",
        "topic": "Attention Mechanism in Transformers",
        "text": """The attention mechanism, introduced in the paper 'Attention Is All You Need' by Vaswani et al. (2017), 
is the core innovation behind the Transformer architecture. Unlike recurrent neural networks that process tokens 
sequentially, attention allows the model to weigh relationships between all tokens in a sequence simultaneously.

The key formula is: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V, where Q (queries), K (keys), and V 
(values) are learned linear projections of the input embeddings. The scaling factor sqrt(d_k) prevents the dot 
products from growing too large, which would push gradients into very small regions of the softmax function.

Multi-head attention extends this by running h parallel attention functions on d_k = d_model/h dimensional 
projections, allowing the model to jointly attend to information from different representation subspaces. 
The outputs are concatenated and projected back to d_model dimensions.

Self-attention (where Q=K=V) allows each position to attend to all other positions in the same sequence. 
This gives Transformers their ability to capture long-range dependencies that RNNs struggle with. The 
computational complexity is O(n^2 * d) per layer, where n is sequence length — this quadratic scaling is 
the main challenge for very long sequences and has motivated research into efficient attention variants like 
Longformer, BigBird, and Flash Attention."""
    },
    {
        "id": "doc_002",
        "topic": "BERT: Bidirectional Encoder Representations",
        "text": """BERT (Bidirectional Encoder Representations from Transformers), published by Devlin et al. at Google 
in 2018, fundamentally changed NLP by introducing deep bidirectional pre-training. Unlike GPT-style models that 
use left-to-right unidirectional attention, BERT uses masked language modeling (MLM) to learn from both left 
and right context simultaneously.

Pre-training objectives: (1) Masked Language Modeling — 15% of tokens are randomly masked, and the model must 
predict the original token. Of masked tokens: 80% are replaced with [MASK], 10% with a random token, and 10% 
are kept unchanged. (2) Next Sentence Prediction (NSP) — the model predicts whether sentence B follows sentence 
A. NSP was later found to be less useful and dropped in RoBERTa.

BERT-Base has 12 transformer layers, 768 hidden dimensions, 12 attention heads, and 110M parameters. 
BERT-Large has 24 layers, 1024 dimensions, 16 heads, and 340M parameters.

Fine-tuning BERT for downstream tasks requires adding a task-specific head on top of the [CLS] token 
representation. BERT achieved state-of-the-art results on 11 NLP benchmarks when released, including 
GLUE, SQuAD, and MultiNLI, demonstrating the power of pre-training on large unlabeled corpora."""
    },
    {
        "id": "doc_003",
        "topic": "GPT Series: Generative Pre-trained Transformers",
        "text": """The GPT (Generative Pre-trained Transformer) series by OpenAI demonstrates the scaling laws of 
language models. GPT-1 (2018) established the pre-train/fine-tune paradigm with 117M parameters. GPT-2 (2019) 
scaled to 1.5B parameters and showed that large models can perform tasks in a zero-shot setting — generating 
coherent long text without task-specific training.

GPT-3 (Brown et al., 2020) with 175B parameters introduced few-shot learning, where providing a few examples 
in the prompt (in-context learning) enables the model to generalize to new tasks without weight updates. The 
paper showed that model capability scales predictably with compute, data, and parameters — described by power 
laws.

GPT-4 (2023) is a multimodal model accepting both text and image inputs. Technical details were not fully 
disclosed, but it uses RLHF (Reinforcement Learning from Human Feedback) for alignment. GPT-4 significantly 
outperforms GPT-3.5 on professional benchmarks like the bar exam (90th percentile) and medical licensing exams.

Key architectural choice in all GPT models: autoregressive (left-to-right) language modeling, where each token 
is predicted conditioned only on preceding tokens. This makes GPT naturally suited for text generation tasks 
compared to BERT's bidirectional approach."""
    },
    {
        "id": "doc_004",
        "topic": "Retrieval-Augmented Generation (RAG)",
        "text": """Retrieval-Augmented Generation (RAG), introduced by Lewis et al. (Facebook AI, 2020), combines 
parametric knowledge stored in model weights with non-parametric knowledge from a retrieved document store. 
This addresses a key limitation of standalone LLMs: factual hallucination and inability to access information 
beyond the training cutoff.

RAG architecture: (1) Indexing — documents are chunked, embedded using a dense retriever (e.g., DPR), and 
stored in a vector database. (2) Retrieval — given a query, the retriever fetches the top-k most similar 
document chunks using Maximum Inner Product Search (MIPS). (3) Generation — retrieved documents are prepended 
to the query as context, and a seq2seq model (e.g., BART) generates the answer.

Two RAG variants: RAG-Sequence uses the same retrieved document for the entire output. RAG-Token allows 
different documents to be retrieved for each output token, enabling more dynamic, multi-document synthesis.

RAG outperforms pure parametric models on knowledge-intensive tasks (Natural Questions, TriviaQA) and allows 
knowledge updates by simply updating the document store without retraining. Key hyperparameters include chunk 
size, overlap, number of retrieved documents (top-k), and embedding model quality. Advanced RAG techniques 
include HyDE (Hypothetical Document Embeddings), reranking with cross-encoders, and recursive retrieval."""
    },
    {
        "id": "doc_005",
        "topic": "Diffusion Models for Image Generation",
        "text": """Diffusion models, formalized by Ho et al. in 'Denoising Diffusion Probabilistic Models' (DDPM, 2020), 
are generative models that learn to reverse a gradual noising process. The forward process adds Gaussian noise 
over T timesteps until the data becomes pure noise. The reverse process trains a neural network to denoise 
iteratively, learning the score function (gradient of log probability).

Training objective: minimize the variational lower bound, which simplifies to predicting the noise added at 
each timestep — a simple MSE loss. The model architecture is typically a U-Net with attention blocks and 
sinusoidal timestep embeddings.

Latent Diffusion Models (LDMs), used in Stable Diffusion (Rombach et al., 2022), perform diffusion in a 
compressed latent space via a pre-trained VAE. This reduces computational cost dramatically while maintaining 
quality. Text conditioning is achieved via cross-attention with CLIP text embeddings.

DALL-E 2 (OpenAI, 2022) and Imagen (Google, 2022) are classifier-free guidance variants where a single model 
is trained with and without conditioning, and outputs are interpolated at inference time. Guidance scale 
controls the tradeoff between sample quality and diversity. DDIM (Song et al., 2020) provides a deterministic 
sampling shortcut reducing inference from 1000 steps to ~50 without retraining."""
    },
    {
        "id": "doc_006",
        "topic": "RLHF: Reinforcement Learning from Human Feedback",
        "text": """Reinforcement Learning from Human Feedback (RLHF), popularized by Christiano et al. (2017) and 
applied at scale in InstructGPT (OpenAI, 2022), aligns language models with human preferences. The three-stage 
process is: (1) Supervised Fine-Tuning (SFT) — fine-tune the base model on high-quality human-written 
demonstrations. (2) Reward Model (RM) training — collect human preference data by ranking model outputs; 
train a classifier to predict which output humans prefer. (3) PPO optimization — use Proximal Policy 
Optimization to optimize the LLM against the reward model, with a KL-divergence penalty to prevent reward 
hacking.

InstructGPT (1.3B parameters) was rated better than GPT-3 (175B) by human evaluators, demonstrating that 
alignment is more valuable than raw scale. The KL penalty coefficient (beta) is crucial: too high and the 
model barely changes; too low and it collapses to reward hacking.

Direct Preference Optimization (DPO, Rafailov et al., 2023) bypasses explicit reward modeling by directly 
training on preference pairs using a binary cross-entropy loss derived from the Bradley-Terry preference model. 
DPO is more stable and computationally cheaper than PPO-based RLHF while achieving similar alignment quality. 
Constitutional AI (Anthropic, 2022) extends RLHF with AI-generated critique and revision cycles."""
    },
        {
        "id": "doc_007",
        "topic": "Graph Neural Networks (GNNs)",
        "text": """Graph Neural Networks (GNNs) extend deep learning to graph-structured data, where entities are nodes 
and relationships are edges. The core operation is message passing: each node aggregates information from its 
neighbors, updates its representation, and repeats for K layers, building up receptive fields over the graph.

Graph Convolutional Networks (GCN, Kipf & Welling, 2017): H^(l+1) = σ(D^{-1/2} A_hat D^{-1/2} H^l W^l), 
where A_hat is the adjacency matrix with self-loops. This is a spectral-domain convolution approximation. 
Limitation: transductive — cannot generalize to unseen nodes.

GraphSAGE (Hamilton et al., 2017) introduces inductive learning by sampling a fixed-size neighborhood and 
aggregating (mean/LSTM/pooling). This enables scalable training on large graphs and generalization to new nodes.

Graph Attention Networks (GAT, Veličković et al., 2018) use attention to weight neighbor contributions, 
making aggregation adaptive rather than uniform. Applications include molecular property prediction (drug 
discovery), social network analysis, knowledge graph completion, and recommendation systems.

Key challenges: over-smoothing (deep GNNs make node representations indistinguishable), over-squashing 
(bottleneck in long-range information propagation), and scalability to billion-edge graphs. Mini-batch 
training with neighbor sampling (PyTorch Geometric, DGL) addresses the scalability issue."""
    },
        {
        "id": "doc_008",
        "topic": "Contrastive Learning: SimCLR and MoCo",
        "text": """Contrastive learning is a self-supervised representation learning framework where models learn by 
distinguishing similar (positive) pairs from dissimilar (negative) pairs, without requiring human labels.

SimCLR (Chen et al., Google, 2020) applies two random augmentations to each image, encodes them with a 
shared ResNet, projects to a lower-dimensional space, and minimizes NT-Xent (normalized temperature-scaled 
cross-entropy) loss. Positives are the two augmented views of the same image; negatives are all other images 
in the batch. Key finding: a non-linear projection head, large batch size (4096–8192), and strong augmentations 
(random crop + color jitter + grayscale) are critical for performance.

MoCo (Momentum Contrast, He et al., Facebook, 2020) uses a momentum encoder with an exponential moving 
average of weights and a queue of negatives, enabling large effective batch sizes without requiring 
large GPU memory. MoCo v2 adds SimCLR's projection head and augmentations.

BYOL (Bootstrap Your Own Latent, Grill et al., 2020) surprisingly eliminates the need for negative pairs 
entirely, using only a target network (momentum encoder) and a predictor network. This challenges the 
traditional view that negatives are essential for preventing collapse.

Linear evaluation protocol: freeze the backbone, train only a linear classifier on top. SimCLR achieves 
76.5% top-1 accuracy on ImageNet with ResNet-50, approaching supervised learning performance."""
    },
    {
        "id": "doc_009",
        "topic": "Neural Architecture Search (NAS)",
        "text": """Neural Architecture Search (NAS) automates the design of neural network architectures, replacing 
manual engineering with optimization algorithms. The NAS problem is defined by three components: (1) search 
space — the set of possible architectures (cells, operations, connections); (2) search strategy — how to 
explore the space; (3) performance estimation — how to evaluate candidate architectures without full training.

Early NAS (Zoph & Le, 2017) used reinforcement learning with an RNN controller generating architecture 
configurations. Results were competitive with human-designed architectures but required 800 GPUs for 28 days.

DARTS (Differentiable Architecture Search, Liu et al., 2019) reformulates the discrete search problem as a 
continuous optimization by relaxing the categorical choice over operations to a weighted mixture. Architecture 
weights and model weights are jointly optimized via gradient descent, reducing search cost to 4 GPU-days.

EfficientNet (Tan & Le, Google, 2019) uses a compound scaling method that uniformly scales network depth, 
width, and resolution using a fixed ratio, found via NAS. EfficientNet-B7 achieves 84.3% top-1 ImageNet 
accuracy with 8.4x fewer parameters than the best existing ConvNet.

One-shot NAS methods share weights across all architectures in a supernet, training once and evaluating 
subnets by weight inheritance. This reduces cost to a single training run but introduces the weight coupling 
problem."""
    },
    {
        "id": "doc_010",
        "topic": "Federated Learning",
        "text": """Federated Learning (FL), introduced by McMahan et al. (Google, 2017) in 'Communication-Efficient 
Learning of Deep Networks from Decentralized Data', trains models across distributed devices without sharing 
raw data — only model updates are communicated. This is critical for privacy-sensitive applications like 
healthcare and mobile keyboards.

FedAvg algorithm: (1) Server broadcasts global model to selected clients. (2) Each client performs E epochs 
of SGD on local data. (3) Clients send updated weights to the server. (4) Server aggregates via weighted 
averaging: w_global = Σ (n_k / n) * w_k, where n_k is client k's data size.

Key challenges: (1) Statistical heterogeneity — non-IID data across clients causes client drift. 
FedProx adds a proximal term to keep local models close to the global model. SCAFFOLD corrects for 
client drift using control variates. (2) System heterogeneity — devices have different compute, memory, 
and connectivity. (3) Communication efficiency — each round requires uploading/downloading model weights; 
gradient compression and quantization reduce bandwidth.

Privacy guarantees: Differential Privacy (DP-FedAvg) adds calibrated Gaussian noise to updates before 
aggregation. Secure Aggregation uses cryptographic protocols so the server never sees individual updates. 
FL is deployed in production at Google (Gboard), Apple (Siri), and hospitals."""
    },
     {
        "id": "doc_011",
        "topic": "Vision Transformers (ViT)",
        "text": """Vision Transformer (ViT, Dosovitskiy et al., Google Brain, 2020) applies the Transformer architecture 
directly to image patches, challenging the dominance of convolutional networks in vision. An image of size 
H×W is divided into N patches of size P×P, where N = HW/P^2. Each patch is linearly projected to D 
dimensions and treated as a token, with learnable positional embeddings added.

A learnable [CLS] token is prepended; its final representation is used for classification. ViT uses standard 
Transformer encoder blocks (LayerNorm → Multi-Head Attention → residual + LayerNorm → MLP → residual).

Key finding: ViT requires large-scale pre-training (JFT-300M, 300 million images) to outperform ResNets. 
On smaller datasets, CNNs' inductive biases (translation invariance, locality) give them an advantage. 
ViT-L/16 achieves 88.55% top-1 on ImageNet-21k with pre-training, beating the prior state-of-the-art.

DeiT (Data-efficient Image Transformers, Touvron et al., 2021) shows ViT can be trained on ImageNet alone 
using knowledge distillation from a CNN teacher and strong augmentations. Swin Transformer introduces 
hierarchical feature maps and shifted window attention, combining the strengths of CNNs and ViT for dense 
prediction tasks like detection and segmentation."""
    },
    {
        "id": "doc_012",
        "topic": "Mixture of Experts (MoE) Models",
        "text": """Mixture of Experts (MoE) scales model capacity without proportionally scaling compute by activating 
only a sparse subset of model parameters for each input. The MoE layer replaces a dense FFN in the Transformer 
with N expert FFN networks and a learned gating/routing function that selects the top-k experts per token.

Sparsely-Gated MoE (Shazeer et al., Google, 2017) demonstrated that routing to only 1-2 experts per token 
enables 1000x capacity increase with only a 2x computational overhead. Key challenge: load balancing — 
without regularization, the gating network collapses to always using a few experts (the 'rich get richer' 
problem). Auxiliary load-balancing loss encourages uniform routing.

Switch Transformer (Fedus et al., 2021) simplifies to top-1 routing, reaching 1 trillion parameters 
and showing clear scaling benefits over dense T5 models of equivalent compute. Expert capacity factor 
controls how many tokens each expert can handle; overflow tokens are dropped or passed through a residual.

Mixtral 8x7B (Mistral AI, 2024) applies MoE to open-source models: 8 experts per layer with top-2 
routing, resulting in 46.7B total parameters but only 12.9B active per token. Mixtral matches or 
outperforms Llama 2 70B on most benchmarks at ~6x lower inference cost."""
    }
]

documents = [doc["text"] for doc in raw_docs]
ids = [doc["id"] for doc in raw_docs]
metadatas = [{"topic": doc["topic"]} for doc in raw_docs]

embeddings = embedder.encode(documents).tolist()

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=ids,
    metadatas=metadatas
)

class CapstoneState(TypedDict):
    question: str
    messages: List
    route: str
    retrieved: str
    sources: List
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int

def route_decision(state):
    return state["route"]

def eval_decision(state):
    if state["faithfulness"] < 0.7 and state["eval_retries"] < 2:
        return "answer"
    return "save"

def memory_node(state: CapstoneState) -> dict:
    msgs = state.get("messages", [])
    msgs = msgs + [{"role": "user", "content": state["question"]}]
    if len(msgs) > 6:  # sliding window: keep last 3 turns
        msgs = msgs[-6:]
    return {"messages": msgs}

def router_node(state: CapstoneState) -> dict:
    question = state["question"]
    messages = state.get("messages", [])
    recent   = "; ".join(f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]) or "none"

    prompt = f"""You are a router for a Research Paper Q&A chatbot.

Options: retrieve / memory_only / tool

Recent: {recent}
Question: {question}

Reply with ONE word only."""

    response = llm.invoke(prompt)
    decision = response.content.strip().lower()

    if "memory" in decision:   decision = "memory_only"
    elif "tool" in decision:   decision = "tool"
    else:                      decision = "retrieve"

    return {"route": decision}

def retrieval_node(state: CapstoneState) -> dict:
    q_emb   = embedder.encode([state["question"]]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)
    chunks  = results["documents"][0]
    topics  = [m["topic"] for m in results["metadatas"][0]]
    context = "\n\n---\n\n".join(f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks)))
    return {"retrieved": context, "sources": topics}

def skip_retrieval_node(state: CapstoneState) -> dict:
    return {"retrieved": "", "sources": []}

def tool_node(state: CapstoneState) -> dict:
    """Web search for recent papers or information not in the knowledge base."""
    question = state["question"]
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(question + " research paper", max_results=3))
        if results:
            tool_result = "Web search results:\n" + "\n".join(
                f"- {r['title']}: {r['body'][:200]}" for r in results
            )
        else:
            tool_result = "Web search returned no results for this query."
    except ImportError:
        tool_result = "Web search tool not available (duckduckgo_search not installed). Please install it with: pip install duckduckgo-search"
    except Exception as e:
        tool_result = f"Web search error: {str(e)}. Try rephrasing your question."

    return {"tool_result": tool_result, "search_results": tool_result}

# Node 5: Answer
def answer_node(state: CapstoneState) -> dict:
    question    = state["question"]
    retrieved   = state.get("retrieved", "")
    tool_result = state.get("tool_result", "")
    messages    = state.get("messages", [])
    eval_retries= state.get("eval_retries", 0)

    context_parts = []
    if retrieved:
        context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
    if tool_result:
        context_parts.append(f"WEB SEARCH RESULTS:\n{tool_result}")
    context = "\n\n".join(context_parts)

    if context:
        system_content = f"""You are a Research Paper Q&A assistant helping PhD students and researchers.
Answer using ONLY the information provided in the context below.
Be precise and technical — your users are experts.
Cite the paper name and authors when available from the context.
If the answer is not in the context, say: I don't have that information in my knowledge base. Try asking about a different paper or concept.
Do NOT add information from your training data beyond what is in the context.

{context}"""
    else:
        system_content = """You are a Research Paper Q&A assistant. Answer based on the conversation history.
If you cannot find the relevant information, politely say so."""

    if eval_retries > 0:
        system_content += "\n\nIMPORTANT: Your previous answer did not meet quality standards. Answer using ONLY information explicitly stated in the context above. Do not add anything from general training knowledge."

    lc_msgs = [SystemMessage(content=system_content)]
    for msg in messages[:-1]:
        lc_msgs.append(HumanMessage(content=msg["content"]) if msg["role"] == "user"
                       else AIMessage(content=msg["content"]))
    lc_msgs.append(HumanMessage(content=question))

    response = llm.invoke(lc_msgs)
    return {"answer": response.content}

print("answer_node defined ")

#  Node 6: Eval 
FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES       = 2

def eval_node(state: CapstoneState) -> dict:
    answer   = state.get("answer", "")
    context  = state.get("retrieved", "")[:500]
    retries  = state.get("eval_retries", 0)

    if not context:
        return {"faithfulness": 1.0, "eval_retries": retries + 1}

    prompt = f"""Rate faithfulness: does this answer use ONLY information from the context?
Reply with ONLY a number between 0.0 and 1.0.
1.0 = fully faithful. 0.5 = some hallucination. 0.0 = mostly hallucinated.

Context: {context}
Answer: {answer[:300]}"""

    result = llm.invoke(prompt).content.strip()
    try:
        score = float(result.split()[0].replace(",", "."))
        score = max(0.0, min(1.0, score))
    except:
        score = 0.5

    gate = "" if score >= FAITHFULNESS_THRESHOLD else ""
    print(f"  [eval] Faithfulness: {score:.2f} {gate}")
    return {"faithfulness": score, "eval_retries": retries + 1}


#  Node 7: Save
def save_node(state: CapstoneState) -> dict:
    messages = state.get("messages", [])
    messages = messages + [{"role": "assistant", "content": state["answer"]}]
    return {"messages": messages}


print("eval_node and save_node defined ")

graph = StateGraph(CapstoneState)

graph.add_node("memory", memory_node)
graph.add_node("router", router_node)
graph.add_node("retrieve", retrieval_node)
graph.add_node("skip", skip_retrieval_node)
graph.add_node("tool", tool_node)
graph.add_node("answer", answer_node)
graph.add_node("eval", eval_node)
graph.add_node("save", save_node)

graph.set_entry_point("memory")

graph.add_edge("memory", "router")

graph.add_conditional_edges("router", route_decision, {
    "retrieve": "retrieve",
    "tool": "tool",
    "skip": "skip"
})

graph.add_edge("retrieve", "answer")
graph.add_edge("tool", "answer")
graph.add_edge("skip", "answer")

graph.add_edge("answer", "eval")

graph.add_conditional_edges("eval", eval_decision, {
    "answer": "answer",
    "save": "save"
})

graph.add_edge("save", END)

app = graph.compile(checkpointer=MemorySaver())

def ask(question, thread_id="default"):
    result = app.invoke(
        {"question": question},
        config={"configurable": {"thread_id": thread_id}}
    )
    return result["answer"]

