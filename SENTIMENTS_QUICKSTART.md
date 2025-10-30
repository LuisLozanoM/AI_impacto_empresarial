# 🎓 News Sentiment Analysis Agent - Student Guide

## 📖 What This Notebook Does
Build an AI agent that fetches news articles and analyzes their sentiment using **LOCAL transformer models** - no API keys required for sentiment analysis!

---

## ⚡ Quick Setup (10-15 minutes)

### Step 1: Install Required Packages

```bash
pip install langgraph langchain-core transformers torch requests matplotlib
```

### Step 2: Open the Notebook

1. Open `sentiments_agents.ipynb` in VS Code
2. Select your Python environment (kernel)
3. Run cell 2 to verify setup ✅

### Step 3: First Run

The first time you run the sentiment analysis cell:
- It will download DistilBERT model (~268MB)
- Takes 30-60 seconds
- Subsequent runs are instant!

---

## 🎯 What You'll Learn

### Core Concepts:
1. **LangGraph Workflows**: Build multi-step AI agents
2. **Local AI Models**: Use Hugging Face transformers without APIs
3. **State Management**: Maintain context across operations
4. **Tool Integration**: Combine AI with external APIs
5. **Workflow Visualization**: Understand agent behavior

### The Workflow:
```
START → Fetch News → Analyze Sentiment → Visualize Results → END
```

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Module not found" | Install missing package: `pip install <package>` |
| Slow first run | Normal! Downloading model (~268MB). Subsequent runs are fast |
| Out of memory | Use smaller batch sizes or close other applications |
| No internet | Model download requires internet first time only |

---

## 💡 Key Features

### ✅ Local Sentiment Analysis
- **Model**: DistilBERT (fast, accurate)
- **Size**: ~268MB (one-time download)
- **Speed**: ~50-100 articles/second
- **Privacy**: Everything runs on your machine

### 📊 What You'll Analyze
- News article sentiment (POSITIVE/NEGATIVE)
- Confidence scores for each prediction
- Aggregate sentiment trends
- Time-based sentiment patterns

---

## 🧪 Try This After Completing

1. **Different Models**: Try emotion-detection models
2. **Real News API**: Get a free key from newsapi.org
3. **Multiple Topics**: Compare sentiment across topics
4. **Time Series**: Track sentiment changes over time
5. **Advanced Filtering**: Filter by confidence, source, etc.

---

## 📚 Understanding the Code

### State (Memory)
```python
class NewsState(TypedDict):
    topic: str              # What to search for
    location: str          # Where to search
    time_period: int       # How many days back
    articles: list         # Fetched news articles
    sentiments: list       # Analysis results
    messages: list         # Conversation log
```

### Nodes (Actions)
- **fetch_news**: Gets articles from News API
- **analyze_sentiment**: Uses local transformer
- **aggregate_results**: Combines and formats results

### Edges (Flow)
- **Sequential**: fetch → analyze → aggregate
- **Conditional**: Can add logic-based routing

---

## 🔄 Workflow Comparison

| Aspect | This Notebook | Cloud APIs |
|--------|--------------|------------|
| **Sentiment Model** | Local (DistilBERT) | Cloud (OpenAI, etc.) |
| **API Keys** | Only for news (free) | Required (costs money) |
| **Privacy** | 100% local analysis | Data sent to cloud |
| **Cost** | Free | $0.002-$0.02 per request |
| **Speed** | Fast (local) | Network latency |
| **Setup** | One-time model download | Just API key |

---

## 📖 Model Details

### DistilBERT for Sentiment Analysis
- **Base**: BERT (Bidirectional Encoder Representations from Transformers)
- **Distilled**: 40% smaller, 60% faster than BERT
- **Fine-tuned**: On SST-2 (Stanford Sentiment Treebank)
- **Accuracy**: ~91% on SST-2 test set
- **Use Case**: General sentiment analysis

### Alternative Models to Try:
```python
# For emotions (happy, sad, angry, etc.)
pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base")

# For Twitter/social media
pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# For financial news
pipeline("sentiment-analysis", model="ProsusAI/finbert")
```

---

## 🎓 Learning Objectives

By completing this notebook, you will:

- ✅ Understand LangGraph state management
- ✅ Integrate local AI models in workflows
- ✅ Build multi-step agent pipelines
- ✅ Visualize agent workflows as graphs
- ✅ Combine AI with traditional APIs
- ✅ Handle errors and edge cases
- ✅ Analyze and aggregate results

---

## 💬 Common Questions

**Q: Why use local models instead of ChatGPT?**  
A: Privacy, cost, speed, and learning! Local models are perfect for specific tasks like sentiment analysis.

**Q: Can I use GPU acceleration?**  
A: Yes! Change `device=-1` to `device=0` in the pipeline initialization.

**Q: How accurate is DistilBERT?**  
A: ~91% on standard benchmarks. Good enough for most use cases!

**Q: Can I analyze non-English text?**  
A: Yes! Use multilingual models like `nlptown/bert-base-multilingual-uncased-sentiment`.

**Q: Is this faster than cloud APIs?**  
A: For sentiment analysis - yes! No network latency. For complex reasoning - cloud models are better.

---

## 📧 Next Steps

1. ✅ Complete the notebook
2. 🧪 Try the exercises
3. 🔄 Modify the workflow
4. 🚀 Build your own agent
5. 📚 Explore more models on Hugging Face

---

## 🌟 Bonus: Combining with LLMs

This notebook focuses on local sentiment analysis. You could extend it by:

1. Using Ollama (local LLM) to summarize articles
2. Using LangChain to connect to different LLMs
3. Building a hybrid: local sentiment + LLM reasoning

Check out `langgraph_tutorial_local.ipynb` for examples of using local LLMs with Ollama!

---

**Happy Learning! 🚀**
