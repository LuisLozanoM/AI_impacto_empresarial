# 🔑 How to Get REAL News Articles (Not Mock Data)

## The Problem

The notebook was using **mock/fake data** with hardcoded URLs like `https://example.com/news/...`

This is NOT real news!

## The Solution: NewsAPI.org

NewsAPI.org provides access to **80,000+ news sources worldwide** with a **FREE tier**.

---

## 🚀 Step-by-Step Setup (2 Minutes)

### 1. Get Your FREE API Key

1. **Visit**: https://newsapi.org/register
2. **Fill out the form**:
   - Your name
   - Your email
   - Create a password
   - Select "Individual/Non-Commercial" (for education/learning)
3. **Click "Submit"**
4. **Check your email** and verify your account
5. **Go to your dashboard**: https://newsapi.org/account
6. **Copy your API Key** (it looks like: `1234567890abcdef1234567890abcdef`)

### 2. Add Your API Key to the Notebook

Open `sentiments_agents.ipynb` and find **Cell 10** (the news fetching cell).

Look for this line:
```python
NEWS_API_KEY = None  # Replace with your key: "YOUR_API_KEY_HERE"
```

**Replace it with**:
```python
NEWS_API_KEY = "your_actual_api_key_here"
```

**Example**:
```python
NEWS_API_KEY = "1234567890abcdef1234567890abcdef"
```

### 3. Run the Notebook

Now when you run the workflow, you'll get **REAL news articles** with:
- ✅ Actual headlines from real news sources
- ✅ Real URLs to the original articles
- ✅ Publication dates
- ✅ Source names (CNN, BBC, TechCrunch, etc.)
- ✅ Author information
- ✅ Article descriptions

---

## 📊 What You Get with Free Tier

| Feature | Free Tier | Paid Tier |
|---------|-----------|-----------|
| **Requests/Day** | 100 | Unlimited |
| **News Sources** | 80,000+ | 80,000+ |
| **Historical Data** | Last 30 days | All history |
| **Commercial Use** | ❌ No | ✅ Yes |
| **Price** | FREE | $449/month |

**For education and learning, the free tier is perfect!**

---

## 🧪 Example: Real vs Mock Data

### ❌ Mock Data (Before):
```json
{
  "title": "Artificial Intelligence Innovation Breakthrough in California",
  "link": "https://example.com/news/artificial-intelligence-innovation-breakthrough",
  "source_id": "tech-news"
}
```

### ✅ Real Data (After with API key):
```json
{
  "title": "OpenAI Announces GPT-5 with Revolutionary Capabilities",
  "link": "https://www.wired.com/story/openai-gpt5-announcement/",
  "source_id": "Wired",
  "author": "John Doe",
  "publishedAt": "2025-10-28T14:32:00Z"
}
```

---

## 🔒 Alternative: Use Environment Variable (More Secure)

Instead of putting your API key directly in the notebook, you can set it as an environment variable:

### Windows (PowerShell):
```powershell
$env:NEWS_API_KEY = "your_api_key_here"
```

### Windows (Command Prompt):
```cmd
set NEWS_API_KEY=your_api_key_here
```

### Mac/Linux:
```bash
export NEWS_API_KEY="your_api_key_here"
```

Then in the notebook, just leave:
```python
NEWS_API_KEY = None  # Will automatically use environment variable
```

---

## 📝 How the Code Works

The `fetch_real_news()` function:

1. **Checks for API key**: Looks for your key in the code or environment
2. **Calls NewsAPI**: Sends a request to `https://newsapi.org/v2/everything`
3. **Filters by topic**: Searches for articles matching your query
4. **Filters by date**: Only gets recent articles (last N days)
5. **Returns real data**: Actual headlines, URLs, sources, etc.
6. **Fallback to mock**: If no API key, uses demo data (for teaching)

---

## 🎯 Sample Queries You Can Try

With your API key, try these topics:

```python
# Technology
initial_state = {'topic': 'artificial intelligence', ...}
initial_state = {'topic': 'cryptocurrency', ...}
initial_state = {'topic': 'cybersecurity', ...}

# Business
initial_state = {'topic': 'stock market', ...}
initial_state = {'topic': 'startup funding', ...}

# Science
initial_state = {'topic': 'climate change', ...}
initial_state = {'topic': 'space exploration', ...}

# Entertainment
initial_state = {'topic': 'movie releases', ...}
initial_state = {'topic': 'gaming industry', ...}
```

---

## ⚠️ Important Notes

### Rate Limits
- **100 requests/day** on free tier
- Each workflow execution = 1 request
- If you hit the limit, wait 24 hours or upgrade

### API Key Security
- ❌ **Don't commit** your API key to GitHub
- ✅ **Use environment variables** for production
- ✅ **Use `.gitignore`** to exclude files with keys

### Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| `401 Unauthorized` | Invalid API key | Check you copied it correctly |
| `426 Upgrade Required` | Hit rate limit | Wait or upgrade plan |
| `429 Too Many Requests` | Exceeded 100/day | Wait 24 hours |

---

## 🆘 Troubleshooting

### "No articles found"
- Try a broader search term
- Check your date range isn't too narrow
- Some topics might not have recent news

### "Invalid API key"
- Make sure you copied the entire key
- No spaces before/after the key
- Check you're using quotes: `"your_key"`

### "Still seeing mock data"
- Make sure you set `NEWS_API_KEY` in Cell 10
- Re-run Cell 10 after setting the key
- Restart the kernel and run all cells

---

## 📚 Additional Resources

- **NewsAPI Documentation**: https://newsapi.org/docs
- **Available Sources**: https://newsapi.org/sources
- **Pricing**: https://newsapi.org/pricing
- **Support**: support@newsapi.org

---

## 🎓 For Students

If you're using this notebook for a class:

1. **Get your own API key** (it's free and takes 2 minutes)
2. **Don't share** your API key with others
3. **Use it responsibly** (don't waste requests)
4. **Cite your sources** when using the news data

Your teacher should **NOT** provide a shared API key, as each student should register their own for learning purposes.

---

## 💡 Pro Tip

Want to save API calls while developing?

1. Run the workflow ONCE with your API key
2. Save the results to a JSON file
3. Load from the file for testing
4. Only call the API when you need fresh data

Example:
```python
import json

# Save results
with open('news_cache.json', 'w') as f:
    json.dump(final_state['sentiment_results'], f)

# Load cached results
with open('news_cache.json', 'r') as f:
    cached_results = json.load(f)
```

---

**Happy news hunting! 📰🎉**
