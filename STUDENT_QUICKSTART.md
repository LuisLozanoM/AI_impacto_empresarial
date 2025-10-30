# 🎓 LangGraph Tutorial - Quick Start Guide for Students

## 📖 What You'll Learn
Build an AI agent that can use tools to solve math problems using **LangGraph** and **local AI models** (no API keys needed!).

---

## ⚡ Quick Setup (15-20 minutes)

### Step 1: Install Ollama

**Choose your operating system:**

**Windows (PowerShell):**
```powershell
winget install Ollama.Ollama
```

**Mac (Terminal):**
```bash
brew install ollama
```

**Linux (Terminal):**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Step 2: Restart VS Code
After installing Ollama, **close and reopen VS Code** so it can find the Ollama command.

### Step 3: Start Ollama Service

**Windows (PowerShell):**
```powershell
Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList "serve" -WindowStyle Hidden
```

**Mac/Linux (Terminal):**
```bash
ollama serve &
```

### Step 4: Download the AI Model

Open a **new terminal** and run:
```bash
ollama pull llama3.2:1b
```

⏱️ **Wait for "success"** - This downloads ~1.3GB (5-15 minutes depending on internet speed)

### Step 5: Install Python Packages

In your Python environment, run:
```bash
pip install langgraph langchain-ollama langchain-core langchain-community typing-extensions
```

### Step 6: Open the Notebook

1. Open `langgraph_tutorial_local.ipynb` in VS Code
2. Select your Python environment (kernel)
3. Run cell 2 to verify setup ✅
4. If you see "SUCCESS", run all cells!

---

## 🎯 What the Notebook Teaches

### Core Concepts:
1. **Tools**: Python functions the AI can use
2. **State**: Memory that persists across the agent's execution
3. **Nodes**: Actions the agent can take (call AI, use tools)
4. **Edges**: How nodes connect (workflow)
5. **Routing**: Logic to decide what to do next

### The Workflow:
```
START → AI thinks → Need a tool? 
                    ├─ Yes → Execute tool → AI thinks again
                    └─ No  → Give final answer → END
```

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Model not found" | Run: `ollama list` - Make sure llama3.2:1b is listed. If not, re-run `ollama pull llama3.2:1b` |
| "Connection refused" | Ollama service isn't running. Start it (see Step 3) |
| Import errors | Install missing package: `pip install <package-name>` |
| Ollama not found | Make sure you restarted VS Code after installing Ollama |
| Slow download | Normal! Large model files take time. Check progress in terminal |

---

## 💡 Tips for Success

### While Going Through the Notebook:
- ✅ **Read every markdown cell** - they explain the concepts
- ✅ **Run cells in order** - each builds on the previous
- ✅ **Check the output** - see how the AI thinks
- ✅ **Try the exercises** - they reinforce learning
- ✅ **Experiment** - change values, add tools, break things!

### Understanding the Code:
- **@tool decorator**: Makes a Python function available to the AI
- **State**: Like variables that persist throughout execution
- **Messages**: The conversation between you, the AI, and tools
- **Nodes**: Think of them as "stations" in a workflow
- **Edges**: The paths between stations

---

## 🧪 Try These After Completing the Tutorial

1. **Add a new math tool** (e.g., square root, modulo)
2. **Change the AI's personality** (modify the system message)
3. **Track additional metrics** (add fields to state)
4. **Build a different agent** (string manipulation, data analysis)
5. **Use a different model** (run `ollama pull mistral` then change model name)

---

## 📚 Learn More

- **LangGraph Docs**: https://docs.langchain.com/oss/python/langgraph/
- **More Ollama Models**: https://ollama.ai/library
- **LangChain Guide**: https://python.langchain.com/docs/

---

## ❓ Common Questions

**Q: Can I use this without internet?**  
A: Yes! After downloading the model, everything runs locally.

**Q: Is this free?**  
A: Yes! Ollama and all the models are free and open source.

**Q: How big is the model?**  
A: Llama 3.2 1B is ~1.3GB. Larger models (3B, 7B) are also available.

**Q: Will this work on my laptop?**  
A: Yes! The 1B model is designed to run on regular laptops.

**Q: What if I want better performance?**  
A: Try larger models: `ollama pull llama3.2:3b` or `ollama pull mistral`

---

## 📧 Need Help?

If you encounter issues:
1. Check the troubleshooting section above
2. Make sure all setup steps completed successfully
3. Ask your instructor
4. Check the error message - it often tells you what's wrong!

---

**Happy Learning! 🚀**
