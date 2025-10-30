# LangGraph with Local Models - Quick Guide

## ✅ What Has Been Set Up

1. **Ollama Installed**: Local LLM server running on your machine
2. **Llama 3.2 3B Downloading**: ~2GB model with tool calling support
3. **LangGraph Installed**: Already available in your environment
4. **Complete Tutorial Created**: `langgraph_tutorial_local.ipynb`

## 🎯 The Goal

Replicate the official LangGraph quickstart tutorial using **100% local models** (no API keys required).

## 📋 Tutorial Structure

The new notebook (`langgraph_tutorial_local.ipynb`) contains:

### Step 1: Define Tools and Model
- Uses `ollama:llama3.2:3b` instead of `anthropic:claude-sonnet-4-5`
- Defines calculator tools: `add`, `multiply`, `divide`
- Binds tools to the model

### Step 2: Define State
- `MessagesState` with message history
- Tracks LLM call count

### Step 3: Define Model Node
- LLM decides whether to call tools
- Returns messages and increments call counter

### Step 4: Define Tool Node
- Executes tool calls
- Returns results as `ToolMessage`

### Step 5: Define Routing Logic
- Routes to tool_node if LLM requests tools
- Routes to END if task is complete

### Step 6: Build and Compile Agent
- Creates StateGraph
- Connects nodes with edges
- Compiles the agent

### Step 7: Test the Agent
- Multiple test cases included
- Shows full conversation flow

## 🚀 How to Run

### Current Status
The Llama 3.2 3B model is downloading (~2GB, ~40% complete).

### Once Download Completes:

1. **Open** the new notebook:
   ```
   langgraph_tutorial_local.ipynb
   ```

2. **Select Kernel**: Choose `ai_ml_env` (Python 3.11)

3. **Run All Cells**: Execute cells sequentially (Shift+Enter or Run All)

4. **Test Examples**: Try the provided test cases or create your own

## 🔧 Troubleshooting

### If Model Isn't Downloaded Yet
Wait for the terminal command to complete:
```powershell
ollama pull llama3.2:3b
```

You'll see: `success` when done.

### If Ollama Service Isn't Running
The service was started automatically, but if you restart your computer:
```powershell
Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList "serve" -WindowStyle Hidden
```

### If Model Loading Fails
Check Ollama is responding:
```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list
```

You should see `llama3.2:3b` in the list.

## 🎓 Key Differences from Original Tutorial

| Aspect | Original Tutorial | Local Version |
|--------|------------------|---------------|
| **Model** | Anthropic Claude Sonnet 4.5 | Ollama Llama 3.2 3B |
| **API Key** | Required ($) | Not Required (Free) |
| **Speed** | Fast (cloud) | Good (local) |
| **Privacy** | Data sent to API | 100% Local |
| **Cost** | Pay per token | Free after download |
| **Setup** | API key only | Ollama + model download |

## 📚 What You'll Learn

1. **LangGraph StateGraph API**: Build agent workflows as graphs
2. **Tool Calling**: Let LLMs use Python functions
3. **State Management**: Track conversation and execution
4. **Routing Logic**: Conditional execution flow
5. **Local LLM Integration**: Use Ollama with LangChain

## 🔄 Alternative Models

Once you understand the tutorial, you can swap models easily:

### Better Performance (Larger Models)
```python
model = init_chat_model("ollama:llama3.1:8b", temperature=0)
model = init_chat_model("ollama:qwen2.5:7b", temperature=0)
```

### Faster Inference (Smaller Models)
```python
model = init_chat_model("ollama:llama3.2:1b", temperature=0)
```

### Download Additional Models
```powershell
ollama pull llama3.1:8b
ollama pull mistral
ollama pull qwen2.5:7b
```

## 📁 Files Overview

- `langgraph_tutorial_local.ipynb` - **NEW**: Complete working tutorial
- `tutorial0.ipynb` - Original experimental code (reference only)
- `LANGGRAPH_LOCAL_GUIDE.md` - This guide
- `README_LANGGRAPH.md` - Project-specific notes (if exists)

## ✨ Next Steps After Tutorial

1. **Experiment**: Modify tools, add new functions
2. **Extend**: Add more complex tools (file operations, calculations, etc.)
3. **Build**: Create your own agents for specific tasks
4. **Explore**: Check LangGraph documentation for advanced features
   - Memory/persistence
   - Human-in-the-loop
   - Multi-agent systems
   - Streaming responses

## 🆘 Need Help?

- LangGraph Docs: https://docs.langchain.com/oss/python/langgraph/
- Ollama Docs: https://ollama.ai/
- LangChain Docs: https://python.langchain.com/

## 📝 Notes

- **No Transformers Solution**: While you have Qwen2-0.5B via Transformers, it doesn't support native tool calling. That's why we use Ollama + Llama 3.2 instead.
- **Why Llama 3.2**: It's small enough to run locally (3B parameters) but has full tool calling support.
- **Performance**: First run may be slow as model loads into memory. Subsequent runs will be faster.

---

**Status**: Model downloading, tutorial ready. Once download completes, you're ready to go! 🚀
