import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import re
import requests
from datetime import datetime
import time
from bs4 import BeautifulSoup

# App config
st.set_page_config(page_title="Qforia Pro", layout="wide")

# Simple CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 Qforia Pro</h1>
    <p>Simple Content Analysis & Research Tool</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_usage' not in st.session_state:
    st.session_state.api_usage = {'gemini_calls': 0, 'perplexity_calls': 0}

# Sidebar - API Configuration
st.sidebar.header("🔧 API Setup")

try:
    gemini_key = st.secrets["api_keys"]["GEMINI_API_KEY"]
    perplexity_key = st.secrets["api_keys"]["PERPLEXITY_API_KEY"]
    st.sidebar.success("✅ API Keys loaded")
except:
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
    perplexity_key = st.sidebar.text_input("Perplexity API Key", type="password")

# API Usage
st.sidebar.subheader("📊 Usage")
st.sidebar.metric("Gemini", st.session_state.api_usage['gemini_calls'])
st.sidebar.metric("Perplexity", st.session_state.api_usage['perplexity_calls'])

# Configure APIs
if gemini_key:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
else:
    model = None

# Core Functions
def scrape_url(url):
    """Simple URL scraping"""
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Get main content
        content_selectors = ['article', 'main', '.content', '.post-content', '.entry-content']
        main_content = None
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                main_content = elements[0]
                break
        
        if not main_content:
            main_content = soup.find('body') or soup
        
        text = main_content.get_text(separator='\n', strip=True)
        title = soup.title.string if soup.title else 'No title'
        
        return {'content': text, 'title': title.strip()}, None
    except Exception as e:
        return None, str(e)

def call_gemini(prompt):
    """Call Gemini API"""
    try:
        response = model.generate_content(prompt)
        st.session_state.api_usage['gemini_calls'] += 1
        return response.text
    except Exception as e:
        return f"Error: {e}"

def call_perplexity(query):
    """Call Perplexity API"""
    try:
        headers = {"Authorization": f"Bearer {perplexity_key}", "Content-Type": "application/json"}
        data = {
            "model": "llama-3.1-sonar-large-128k-online",
            "messages": [{"role": "user", "content": query}],
            "temperature": 0.2,
            "max_tokens": 1000
        }
        
        response = requests.post("https://api.perplexity.ai/chat/completions", 
                               headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            st.session_state.api_usage['perplexity_calls'] += 1
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error: API call failed"
    except Exception as e:
        return f"Error: {e}"

def extract_primary_keyword(content):
    """Extract primary keyword using Gemini"""
    prompt = f"""
    Analyze this content and identify the PRIMARY KEYWORD (1-3 words) that best represents the main topic.
    
    Content: {content[:2000]}...
    
    Return only the primary keyword, nothing else.
    """
    return call_gemini(prompt).strip().replace('"', '')

def qforia_analysis(keyword):
    """Generate Qforia fan-out queries"""
    prompt = f"""
    Generate 15-20 research queries for comprehensive analysis of: "{keyword}"

    Include these query types:
    1. Market data & statistics
    2. Recent trends & developments  
    3. Investment information
    4. Comparative analysis
    5. Financial data
    6. Growth metrics

    Return as JSON array: ["query1", "query2", ...]
    """
    
    try:
        response = call_gemini(prompt)
        # Clean JSON response
        json_text = response.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        
        queries = json.loads(json_text.strip())
        return queries if isinstance(queries, list) else []
    except:
        return []

def identify_missing_context(content, keyword):
    """Identify missing context and data points"""
    prompt = f"""
    Analyze this content about "{keyword}" and identify missing information.
    
    Content: {content[:3000]}...
    
    Return JSON:
    {
        "missing_context": [
            {"topic": "Missing Topic", "importance": "high/medium/low", "description": "What's missing"}
        ],
        "missing_data": [
            {"type": "Statistics/Financial/Market", "description": "What data is missing"}
        ]
    }
    """
    
    try:
        response = call_gemini(prompt)
        json_text = response.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        
        return json.loads(json_text.strip())
    except:
        return {"missing_context": [], "missing_data": []}

def fact_check_content(content):
    """Fact-check content using Perplexity"""
    # Extract key claims from content
    prompt = f"""
    Extract 5-7 key factual claims from this content that can be verified:
    
    {content[:2000]}...
    
    Return as simple list: ["claim1", "claim2", ...]
    """
    
    try:
        response = call_gemini(prompt)
        json_text = response.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        if json_text.startswith("["):
            claims = json.loads(json_text)
        else:
            # Fallback: split by lines
            claims = [line.strip().strip('"') for line in response.split('\n') if line.strip()]
        
        # Verify each claim
        fact_check_results = []
        for claim in claims[:5]:  # Limit to 5 to control costs
            if len(claim) > 10:  # Only check substantial claims
                verification = call_perplexity(f"Verify this claim with current data: {claim}")
                fact_check_results.append({
                    'claim': claim,
                    'verification': verification[:300] + "..." if len(verification) > 300 else verification
                })
        
        return fact_check_results
    except:
        return []

# Main Interface
feature = st.selectbox(
    "Choose Feature:",
    ["🔗 URL Analysis", "🔍 Keyword Research", "📊 Direct Research", "✅ Fact Checker"]
)

st.markdown("---")

# Feature 1: URL Analysis
if feature == "🔗 URL Analysis":
    st.header("🔗 URL Content Analysis")
    
    url = st.text_input("Enter URL:", placeholder="https://example.com/article")
    
    if st.button("Analyze URL") and url and model:
        with st.spinner("Analyzing URL..."):
            # Step 1: Scrape URL
            scraped_data, error = scrape_url(url)
            
            if scraped_data:
                content = scraped_data['content']
                title = scraped_data['title']
                
                st.success(f"✅ Scraped: {title}")
                
                # Step 2: Extract primary keyword
                primary_keyword = extract_primary_keyword(content)
                st.info(f"🎯 Primary Keyword: **{primary_keyword}**")
                
                # Step 3: Identify missing context
                missing_analysis = identify_missing_context(content, primary_keyword)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📋 Missing Context")
                    if missing_analysis.get('missing_context'):
                        missing_df = pd.DataFrame(missing_analysis['missing_context'])
                        st.dataframe(missing_df, hide_index=True)
                    else:
                        st.info("No missing context identified")
                
                with col2:
                    st.subheader("📊 Missing Data Points")
                    if missing_analysis.get('missing_data'):
                        data_df = pd.DataFrame(missing_analysis['missing_data'])
                        st.dataframe(data_df, hide_index=True)
                    else:
                        st.info("No missing data points identified")
                
                # Step 4: Generate research topics
                st.subheader("🔍 Research Topics")
                research_queries = qforia_analysis(primary_keyword)
                
                if research_queries:
                    selected_queries = st.multiselect(
                        "Select topics to research:",
                        research_queries,
                        default=research_queries[:5]
                    )
                    
                    if st.button("🚀 Research Selected Topics") and selected_queries:
                        research_results = []
                        progress = st.progress(0)
                        
                        for i, query in enumerate(selected_queries):
                            progress.progress((i + 1) / len(selected_queries))
                            result = call_perplexity(query)
                            research_results.append({'Query': query, 'Research': result[:200] + "..."})
                            time.sleep(0.5)
                        
                        st.subheader("📊 Research Results")
                        results_df = pd.DataFrame(research_results)
                        st.dataframe(results_df, hide_index=True)
            else:
                st.error(f"Failed to scrape URL: {error}")

# Feature 2: Keyword Research  
elif feature == "🔍 Keyword Research":
    st.header("🔍 Keyword Research")
    
    keyword = st.text_input("Enter Keyword/Topic:", placeholder="e.g., Bangalore Real Estate")
    
    if st.button("Generate Research Topics") and keyword and model:
        with st.spinner("Generating research topics..."):
            queries = qforia_analysis(keyword)
            
            if queries:
                st.success(f"✅ Generated {len(queries)} research topics")
                
                selected_queries = st.multiselect(
                    "Select topics to research:",
                    queries,
                    default=queries[:5]
                )
                
                if st.button("🚀 Research Selected") and selected_queries:
                    research_results = []
                    progress = st.progress(0)
                    
                    for i, query in enumerate(selected_queries):
                        progress.progress((i + 1) / len(selected_queries))
                        result = call_perplexity(query)
                        research_results.append({'Query': query, 'Research': result[:200] + "..."})
                        time.sleep(0.5)
                    
                    st.subheader("📊 Research Results")
                    results_df = pd.DataFrame(research_results)
                    st.dataframe(results_df, hide_index=True)

# Feature 3: Direct Research
elif feature == "📊 Direct Research":
    st.header("📊 Direct Topic Research")
    
    topic = st.text_input("Enter Research Topic:", placeholder="e.g., Latest AI market trends 2024")
    
    if st.button("Research Topic") and topic:
        with st.spinner("Researching..."):
            result = call_perplexity(topic)
            
            st.subheader("📋 Research Result")
            st.markdown(result)

# Feature 4: Fact Checker
elif feature == "✅ Fact Checker":
    st.header("✅ URL Fact Checker")
    
    url = st.text_input("Enter URL to fact-check:", placeholder="https://example.com/article")
    
    if st.button("Fact Check URL") and url and model:
        with st.spinner("Fact-checking..."):
            # Scrape URL
            scraped_data, error = scrape_url(url)
            
            if scraped_data:
                content = scraped_data['content']
                title = scraped_data['title']
                
                st.success(f"✅ Analyzing: {title}")
                
                # Fact check
                fact_results = fact_check_content(content)
                
                if fact_results:
                    st.subheader("📊 Fact Check Report")
                    fact_df = pd.DataFrame(fact_results)
                    st.dataframe(fact_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No factual claims identified for verification")
            else:
                st.error(f"Failed to scrape URL: {error}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🗑️ Clear Data"):
        st.session_state.api_usage = {'gemini_calls': 0, 'perplexity_calls': 0}
        st.success("Data cleared!")

with col2:
    total_calls = st.session_state.api_usage['gemini_calls'] + st.session_state.api_usage['perplexity_calls']
    st.metric("Total API Calls", total_calls)

with col3:
    cost = (st.session_state.api_usage['perplexity_calls'] * 0.002) + (st.session_state.api_usage['gemini_calls'] * 0.001)
    st.metric("Estimated Cost", f"${cost:.3f}")

st.markdown("**Qforia Pro** - Simple Content Analysis & Research Tool")
