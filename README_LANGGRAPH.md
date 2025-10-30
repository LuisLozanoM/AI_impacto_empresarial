# 📰 LangGraph News Sentiment Analysis Tutorial

A comprehensive tutorial demonstrating how to build a **LangGraph workflow** that retrieves news articles and analyzes their sentiment using local AI models.

## 🎯 Overview

This project showcases:
- **LangGraph** for building stateful AI workflows
- **News retrieval** using free APIs (with mock data fallback)
- **Sentiment analysis** using local transformer models (no API keys needed!)
- **Interactive visualization** with Streamlit
- **Jupyter Notebook** tutorial with step-by-step explanations

## 📁 Project Structure

```
├── snetiments_agents.ipynb    # Complete LangGraph tutorial notebook
├── news_sentiment_app.py      # Streamlit web application
├── README_LANGGRAPH.md        # This file
└── ai_ml_env/                 # Virtual environment
```

## 🚀 Quick Start

### 1. Prerequisites

The required packages are already installed in your `ai_ml_env` virtual environment:
- langgraph
- langchain
- transformers
- streamlit
- newsapi-python
- pandas
- matplotlib

### 2. Run the Jupyter Notebook Tutorial

Open `snetiments_agents.ipynb` in VS Code or Jupyter Lab and run the cells sequentially to learn about LangGraph step by step.

The notebook covers:
1. **State Definition** - Creating a shared data structure
2. **Tools** - Building news retrieval and sentiment analysis tools
3. **Nodes** - Defining workflow steps
4. **Graph Construction** - Connecting nodes with edges
5. **Visualization** - Viewing the workflow graph
6. **Execution** - Running the complete workflow
7. **Results** - Analyzing and visualizing sentiment data

### 3. Run the Streamlit App

Launch the interactive web application:

```powershell
streamlit run news_sentiment_app.py
```

Or with the full Python path:

```powershell
C:/Users/L03055876/Desktop/AI_impacto_empresarial/ai_ml_env/Scripts/python.exe -m streamlit run news_sentiment_app.py
```

The app will open in your browser at `http://localhost:8501`

## 🏗️ Architecture

### LangGraph Workflow

```
START
  ↓
┌─────────────────────┐
│  retrieve_news      │ ← Fetch articles based on topic, location, time
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ analyze_sentiment   │ ← Analyze each article using local AI model
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│    summarize        │ ← Generate summary statistics
└──────────┬──────────┘
           ↓
         END
```

### State Object

The `AgentState` flows through all nodes and contains:

```python
{
    'topic': str,              # News topic to search
    'location': str,           # Geographic location
    'time_interval': int,      # Days to look back
    'news_articles': list,     # Retrieved articles
    'sentiment_results': list, # Analysis results
    'messages': list           # Conversation history
}
```

## 🛠️ Key Features

### 1. News Retrieval Tool

Currently uses **mock data** for demonstration. To use real news:

**Option A: NewsAPI (100 requests/day free)**
- Get API key: https://newsapi.org/
- Replace the fetch function in the code

**Option B: NewsData.io (200 requests/day free)**
- Get API key: https://newsdata.io/
- Already integrated, just add your API key

### 2. Sentiment Analysis

Uses **DistilBERT** model fine-tuned for sentiment analysis:
- Runs completely **locally** (no API needed)
- Fast inference on CPU
- High accuracy for English text
- Returns sentiment label and confidence score

### 3. Streamlit Interface

Interactive web UI with:
- **Configuration panel** - Adjust topic, location, time range
- **Real-time execution** - Watch the workflow progress
- **Visual analytics** - Pie charts and bar graphs
- **Detailed results** - Complete data tables
- **Article viewer** - Read individual articles with sentiment

## 📊 Example Usage

### From Notebook

```python
# Define your search parameters
initial_state = {
    'topic': 'artificial intelligence',
    'location': 'California',
    'time_interval': 7,
    'news_articles': [],
    'sentiment_results': [],
    'messages': []
}

# Run the workflow
final_state = app.invoke(initial_state)

# View results
for result in final_state['sentiment_results']:
    print(f"{result['article']}: {result['sentiment']} ({result['confidence']:.2%})")
```

### From Streamlit

1. Enter your search topic (e.g., "climate change")
2. Specify location (e.g., "New York")
3. Set time range (e.g., 7 days)
4. Click "Analyze News"
5. Explore results in different tabs

## 🧩 Extending the Workflow

### Add Conditional Routing

```python
def should_continue(state: AgentState) -> str:
    """Route based on article count"""
    if len(state['news_articles']) > 10:
        return "analyze_detailed"
    else:
        return "analyze_simple"

workflow.add_conditional_edges(
    "retrieve_news",
    should_continue,
    {
        "analyze_detailed": "detailed_analysis",
        "analyze_simple": "simple_analysis"
    }
)
```

### Add Human-in-the-Loop

```python
def approval_node(state: AgentState) -> AgentState:
    """Pause for human approval"""
    print(f"Found {len(state['news_articles'])} articles. Proceed?")
    approval = input("y/n: ")
    
    if approval.lower() != 'y':
        state['messages'].append(HumanMessage(content="Workflow cancelled by user"))
        return END
    
    return state
```

### Add More Analysis Tools

```python
# Topic modeling
def extract_topics_node(state: AgentState) -> AgentState:
    """Extract main topics from articles"""
    # Use NLP models to identify topics
    return state

# Entity extraction
def extract_entities_node(state: AgentState) -> AgentState:
    """Extract named entities (people, organizations, locations)"""
    # Use NER models
    return state
```

## 📚 Learning Resources

### LangGraph
- [Official Documentation](https://langchain-ai.github.io/langgraph/)
- [GitHub Examples](https://github.com/langchain-ai/langgraph/tree/main/examples)
- [Tutorial Videos](https://python.langchain.com/docs/langgraph)

### Transformers
- [Hugging Face Course](https://huggingface.co/course)
- [Sentiment Analysis Guide](https://huggingface.co/tasks/text-classification)

### Streamlit
- [Documentation](https://docs.streamlit.io/)
- [Gallery](https://streamlit.io/gallery)

## 🎓 Key Concepts Explained

### What is LangGraph?

LangGraph is a framework for building **stateful, multi-actor applications** with LLMs. Unlike simple chains, it allows:
- **Cycles** - Workflows can loop back
- **Persistence** - State is maintained across steps
- **Branching** - Conditional routing based on data
- **Human-in-the-loop** - Pause for human input

### Why Use LangGraph?

1. **Complex Workflows** - Multi-step processes with decision points
2. **State Management** - Share data between steps cleanly
3. **Debugging** - Visualize and inspect workflow execution
4. **Scalability** - Easy to add new nodes and edges
5. **Integration** - Works with LangChain tools and agents

### Nodes vs Edges

- **Nodes**: Functions that transform state
  - Take current state as input
  - Perform actions (API calls, computations)
  - Return updated state

- **Edges**: Define execution flow
  - **Simple edges**: Always go to next node
  - **Conditional edges**: Choose path based on state
  - **END edge**: Terminate workflow

## 🔧 Troubleshooting

### Model Loading Issues

If the sentiment model fails to load:
```python
# Try clearing cache
import shutil
shutil.rmtree("~/.cache/huggingface/hub", ignore_errors=True)
```

### Memory Issues

For large datasets:
```python
# Process in batches
def analyze_in_batches(articles, batch_size=10):
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i+batch_size]
        # Process batch
```

### Streamlit Port Already in Use

```powershell
streamlit run news_sentiment_app.py --server.port 8502
```

## 📈 Future Enhancements

- [ ] Real-time news streaming
- [ ] Multiple language support
- [ ] Comparative analysis across time periods
- [ ] Export results to PDF/CSV
- [ ] Email notifications for specific sentiment patterns
- [ ] Integration with more news sources
- [ ] Advanced NLP features (topic modeling, entity recognition)
- [ ] Dashboard for historical trends

## 📝 License

This project is for educational purposes. Feel free to use and modify!

## 🤝 Contributing

Suggestions and improvements are welcome! Key areas:
- Additional news APIs integration
- More sophisticated sentiment models
- Enhanced visualizations
- Performance optimizations

## 📞 Support

For questions about:
- **LangGraph**: Check the [official docs](https://langchain-ai.github.io/langgraph/)
- **Transformers**: Visit [Hugging Face forums](https://discuss.huggingface.co/)
- **This project**: Review the notebook tutorial for detailed explanations

---

**Happy Learning! 🎉**

Built with ❤️ using LangGraph, Transformers, and Streamlit
