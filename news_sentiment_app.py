"""
News Sentiment Analysis App using LangGraph
A Streamlit application that retrieves news and analyzes sentiment using local AI models
"""

import streamlit as st
from typing import TypedDict, Annotated, Sequence
from datetime import datetime, timedelta
import operator
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage

# Configure Streamlit page
st.set_page_config(
    page_title="News Sentiment Analyzer",
    page_icon="📰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """State object that flows through the graph"""
    topic: str
    location: str
    time_interval: int
    news_articles: list
    sentiment_results: list
    messages: Annotated[Sequence[BaseMessage], operator.add]


# ============================================================================
# TOOLS - NEWS RETRIEVAL
# ============================================================================

def fetch_news_free(topic: str, location: str, days_back: int) -> list:
    """
    Fetch news articles (using mock data for demo)
    In production, replace with real API like NewsAPI or NewsData.io
    """
    # Mock articles for demonstration
    mock_articles = [
        {
            'title': f'{topic.title()} Innovation Breakthrough in {location}',
            'description': f'New developments in {topic} technology are revolutionizing the industry, bringing significant changes to {location}.',
            'pubDate': (datetime.now() - timedelta(days=1)).isoformat(),
            'source_id': 'tech-news'
        },
        {
            'title': f'Major {topic.title()} Conference Held in {location}',
            'description': f'Industry leaders gathered to discuss the future of {topic}, with experts predicting major growth in the {location} region.',
            'pubDate': (datetime.now() - timedelta(days=2)).isoformat(),
            'source_id': 'business-insider'
        },
        {
            'title': f'{topic.title()} Market Faces Challenges',
            'description': f'Despite recent setbacks, analysts remain optimistic about {topic} prospects in {location} for the coming year.',
            'pubDate': (datetime.now() - timedelta(days=3)).isoformat(),
            'source_id': 'market-watch'
        },
        {
            'title': f'New {topic.title()} Startup Raises Funding in {location}',
            'description': f'A promising {topic} startup secured significant investment to expand operations across {location}.',
            'pubDate': (datetime.now() - timedelta(days=4)).isoformat(),
            'source_id': 'venture-beat'
        },
        {
            'title': f'{topic.title()} Regulations Updated in {location}',
            'description': f'Local authorities in {location} announced new regulations affecting the {topic} sector, sparking debate among stakeholders.',
            'pubDate': (datetime.now() - timedelta(days=5)).isoformat(),
            'source_id': 'policy-news'
        }
    ]
    
    return mock_articles


# ============================================================================
# SENTIMENT ANALYSIS
# ============================================================================

@st.cache_resource
def load_sentiment_model():
    """Load sentiment analysis model (cached)"""
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )


def analyze_sentiment(text: str, analyzer) -> dict:
    """Analyze sentiment of text"""
    try:
        result = analyzer(text[:512])[0]
        return {
            'label': result['label'],
            'score': result['score'],
            'sentiment': 'positive' if result['label'] == 'POSITIVE' else 'negative'
        }
    except Exception as e:
        return {'label': 'NEUTRAL', 'score': 0.5, 'sentiment': 'neutral'}


# ============================================================================
# LANGGRAPH NODES
# ============================================================================

def retrieve_news_node(state: AgentState) -> AgentState:
    """Node 1: Retrieve news articles"""
    with st.spinner("📰 Retrieving news articles..."):
        articles = fetch_news_free(
            topic=state['topic'],
            location=state['location'],
            days_back=state['time_interval']
        )
        
        state['news_articles'] = articles
        state['messages'].append(
            HumanMessage(content=f"Retrieved {len(articles)} articles")
        )
    
    return state


def analyze_sentiment_node(state: AgentState) -> AgentState:
    """Node 2: Analyze sentiment"""
    with st.spinner("🎭 Analyzing sentiment..."):
        analyzer = load_sentiment_model()
        results = []
        
        progress_bar = st.progress(0)
        articles = state['news_articles']
        
        for i, article in enumerate(articles):
            text = f"{article.get('title', '')} {article.get('description', '')}"
            sentiment = analyze_sentiment(text, analyzer)
            
            results.append({
                'article': article.get('title', 'Unknown'),
                'description': article.get('description', ''),
                'sentiment': sentiment['sentiment'],
                'confidence': sentiment['score'],
                'date': article.get('pubDate', 'Unknown'),
                'source': article.get('source_id', 'Unknown')
            })
            
            progress_bar.progress((i + 1) / len(articles))
        
        progress_bar.empty()
        state['sentiment_results'] = results
        state['messages'].append(
            HumanMessage(content=f"Analyzed {len(results)} articles")
        )
    
    return state


def summarize_results_node(state: AgentState) -> AgentState:
    """Node 3: Summarize results"""
    results = state['sentiment_results']
    
    if results:
        positive = sum(1 for r in results if r['sentiment'] == 'positive')
        negative = sum(1 for r in results if r['sentiment'] == 'negative')
        
        summary = f"Analyzed {len(results)} articles: {positive} positive, {negative} negative"
        state['messages'].append(HumanMessage(content=summary))
    
    return state


# ============================================================================
# BUILD LANGGRAPH
# ============================================================================

@st.cache_resource
def build_workflow():
    """Build and compile the LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve_news", retrieve_news_node)
    workflow.add_node("analyze_sentiment", analyze_sentiment_node)
    workflow.add_node("summarize", summarize_results_node)
    
    # Define edges
    workflow.set_entry_point("retrieve_news")
    workflow.add_edge("retrieve_news", "analyze_sentiment")
    workflow.add_edge("analyze_sentiment", "summarize")
    workflow.add_edge("summarize", END)
    
    return workflow.compile()


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📰 News Sentiment Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by LangGraph & Local AI Models")
    
    # Sidebar - Input Parameters
    st.sidebar.header("⚙️ Configuration")
    
    topic = st.sidebar.text_input(
        "News Topic",
        value="artificial intelligence",
        help="Enter the topic you want to search for"
    )
    
    location = st.sidebar.text_input(
        "Location",
        value="California",
        help="Enter the location/state"
    )
    
    time_interval = st.sidebar.slider(
        "Time Range (days)",
        min_value=1,
        max_value=30,
        value=7,
        help="Number of days to look back"
    )
    
    analyze_button = st.sidebar.button("🚀 Analyze News", type="primary", use_container_width=True)
    
    # Information boxes
    with st.sidebar.expander("ℹ️ About This App"):
        st.write("""
        This app demonstrates a **LangGraph workflow** that:
        - Retrieves news articles
        - Analyzes sentiment using local AI models
        - Visualizes results
        
        **Technologies:**
        - LangGraph for workflow orchestration
        - Transformers for sentiment analysis
        - Streamlit for UI
        """)
    
    with st.sidebar.expander("📊 Workflow Graph"):
        st.code("""
        START
          ↓
        retrieve_news
          ↓
        analyze_sentiment
          ↓
        summarize
          ↓
        END
        """)
    
    # Main content
    if analyze_button:
        # Build workflow
        app = build_workflow()
        
        # Prepare initial state
        initial_state = {
            'topic': topic,
            'location': location,
            'time_interval': time_interval,
            'news_articles': [],
            'sentiment_results': [],
            'messages': []
        }
        
        # Execute workflow
        st.info(f"🔍 Analyzing news about **{topic}** in **{location}** from the last **{time_interval} days**")
        
        try:
            final_state = app.invoke(initial_state)
            results = final_state['sentiment_results']
            
            if results:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                positive = sum(1 for r in results if r['sentiment'] == 'positive')
                negative = sum(1 for r in results if r['sentiment'] == 'negative')
                avg_confidence = sum(r['confidence'] for r in results) / len(results)
                
                with col1:
                    st.metric("Total Articles", len(results))
                with col2:
                    st.metric("Positive", positive, f"{positive/len(results)*100:.1f}%")
                with col3:
                    st.metric("Negative", negative, f"{negative/len(results)*100:.1f}%")
                with col4:
                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                
                # Tabs for different views
                tab1, tab2, tab3 = st.tabs(["📊 Visualizations", "📋 Detailed Results", "📰 Articles"])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Pie chart
                        st.subheader("Sentiment Distribution")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        sentiment_counts = {}
                        for r in results:
                            s = r['sentiment']
                            sentiment_counts[s] = sentiment_counts.get(s, 0) + 1
                        
                        colors = {'positive': '#4CAF50', 'negative': '#F44336', 'neutral': '#FFC107'}
                        pie_colors = [colors.get(s, '#999') for s in sentiment_counts.keys()]
                        
                        ax.pie(
                            sentiment_counts.values(),
                            labels=[s.upper() for s in sentiment_counts.keys()],
                            autopct='%1.1f%%',
                            colors=pie_colors,
                            startangle=90
                        )
                        st.pyplot(fig)
                    
                    with col2:
                        # Bar chart
                        st.subheader("Confidence Scores")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        articles_labels = [f"A{i+1}" for i in range(len(results))]
                        confidences = [r['confidence'] for r in results]
                        bar_colors = [colors.get(r['sentiment'], '#999') for r in results]
                        
                        ax.bar(articles_labels, confidences, color=bar_colors)
                        ax.set_xlabel('Articles')
                        ax.set_ylabel('Confidence Score')
                        ax.set_ylim(0, 1)
                        st.pyplot(fig)
                
                with tab2:
                    # Data table
                    st.subheader("Detailed Sentiment Analysis")
                    df = pd.DataFrame(results)
                    df['confidence'] = df['confidence'].apply(lambda x: f"{x:.2%}")
                    df = df[['article', 'sentiment', 'confidence', 'date', 'source']]
                    df.index = df.index + 1
                    st.dataframe(df, use_container_width=True)
                
                with tab3:
                    # Article cards
                    st.subheader("News Articles")
                    for i, result in enumerate(results):
                        sentiment = result['sentiment']
                        emoji = "✅" if sentiment == 'positive' else "❌"
                        color = "green" if sentiment == 'positive' else "red"
                        
                        with st.expander(f"{emoji} {result['article']}"):
                            st.markdown(f"**Sentiment:** :{color}[{sentiment.upper()}]")
                            st.markdown(f"**Confidence:** {result['confidence']:.2%}")
                            st.markdown(f"**Source:** {result['source']}")
                            st.markdown(f"**Date:** {result['date'][:10]}")
                            st.markdown(f"**Description:** {result['description']}")
                
                st.success("✅ Analysis completed successfully!")
            else:
                st.warning("No articles found for the given criteria.")
        
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
    
    else:
        # Welcome message
        st.info("👈 Configure your search parameters in the sidebar and click **Analyze News** to start!")
        
        # Example workflow visualization
        st.subheader("🔄 LangGraph Workflow")
        st.image("https://python.langchain.com/v0.1/assets/images/langgraph-8e6d4e6be29899e8d1a0cd6c2ce57066.png", 
                 caption="LangGraph enables building stateful, multi-step AI workflows")


if __name__ == "__main__":
    main()
