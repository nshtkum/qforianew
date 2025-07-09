import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import requests
from datetime import datetime
from bs4 import BeautifulSoup
import time

# App config
st.set_page_config(page_title="Qforia Simplified", layout="wide")
st.title("🔍 Qforia: URL Content Analysis & Fact Checker")

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'research_data' not in st.session_state:
    st.session_state.research_data = []

# Sidebar for API keys
st.sidebar.header("🔧 Configuration")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
perplexity_key = st.sidebar.text_input("Perplexity API Key", type="password")

# Configure Gemini
def setup_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        st.sidebar.error(f"Gemini setup failed: {e}")
        return None

model = setup_gemini(gemini_key) if gemini_key else None

# Scrape URL content
def scrape_url(url):
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for element in soup(['script', 'style', 'nav', 'footer']):
            element.decompose()
        
        content = soup.find('article') or soup.find('main') or soup.find('body')
        text = content.get_text(separator=' ', strip=True) if content else ""
        
        if len(text) < 100:
            return None, "Content too short"
        
        title = soup.title.string.strip() if soup.title else "No title"
        return {'content': text, 'title': title, 'url': url}, None
    except Exception as e:
        return None, f"Scraping error: {e}"

# Perplexity API call
def call_perplexity(query):
    if not perplexity_key:
        return {"error": "Missing Perplexity API key"}
    
    headers = {
        "Authorization": f"Bearer {perplexity_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    data = {
        "model": "sonar-medium-online",  # Updated to supported model (April 2025)
        "messages": [
            {"role": "system", "content": "Provide factual, concise information with recent data."},
            {"role": "user", "content": query}
        ],
        "max_tokens": 500
    }
    
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=data,
            timeout=15
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        try:
            error_detail = response.json().get('error', {}).get('message', str(e))
        except:
            error_detail = str(e)
        return {"error": f"Perplexity API error: {response.status_code} - {error_detail}"}
    except Exception as e:
        return {"error": f"Perplexity API error: {e}"}

# Analyze topics with Gemini
def analyze_topics(content, keyword=""):
    if not model:
        return None, "Gemini not configured"
    
    try:
        prompt = f"""
        Analyze the content and identify main topics and missing context.
        Primary keyword: {keyword}
        Content: {content[:4000]}

        Return JSON:
        {{
            "main_topics": ["topic1", "topic2"],
            "missing_context": [
                {{"topic": "topic", "missing_info": "description", "research_query": "query"}}
            ]
        }}
        """
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:-3]
        return json.loads(json_text), None
    except Exception as e:
        return None, f"Analysis error: {e}"

# Research missing context
def research_missing_context(missing_items):
    results = []
    for item in missing_items[:5]:
        query = item.get('research_query', f"Latest data on {item['topic']} {item['missing_info']}")
        time.sleep(1)  # Delay to avoid rate limits
        response = call_perplexity(query)
        
        if 'choices' in response:
            content = response['choices'][0]['message']['content']
            results.append({
                'topic': item['topic'],
                'missing_info': item['missing_info'],
                'research_content': content,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
            })
        else:
            results.append({
                'topic': item['topic'],
                'missing_info': item['missing_info'],
                'research_content': f"Error: {response.get('error', 'Unknown')}",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
            })
    return results

# Main URL Analysis
st.subheader("🌐 Analyze URL")
url = st.text_input("Enter URL:", placeholder="https://example.com")
keyword = st.text_input("Primary Keyword (Optional):", placeholder="e.g., technology")
if st.button("Analyze", disabled=not (url and gemini_key and perplexity_key)):
    with st.spinner("Analyzing..."):
        # Scrape content
        scraped_data, error = scrape_url(url)
        if scraped_data:
            st.success("Content scraped!")
            st.write(f"**Title:** {scraped_data['title']}")
            
            # Analyze topics
            analysis, error = analyze_topics(scraped_data['content'], keyword)
            if analysis:
                st.session_state.analysis_results = analysis
                st.write("**Main Topics:**")
                for topic in analysis.get('main_topics', []):
                    st.write(f"- {topic}")
                
                # Research missing context
                missing_context = analysis.get('missing_context', [])
                if missing_context:
                    st.session_state.research_data = research_missing_context(missing_context)
                    st.success("Research completed!")
                else:
                    st.info("No missing context identified.")
            else:
                st.error(f"Analysis failed: {error}")
        else:
            st.error(f"Scraping failed: {error}")

# Display Research Results
if st.session_state.research_data:
    st.subheader("📊 Research Results")
    df_data = [{
        'Topic': r['topic'],
        'Missing Info': r['missing_info'],
        'Research Findings': r['research_content'][:200] + "..." if len(r['research_content']) > 200 else r['research_content'],
        'Timestamp': r['timestamp']
    } for r in st.session_state.research_data]
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)

# Fact Checker
st.subheader("🔍 Fact Checker")
fact_query = st.text_input("Enter claim to verify:", placeholder="e.g., Smartphone market grew 12% in 2024")
if st.button("Verify Fact", disabled=not (fact_query and perplexity_key)):
    with st.spinner("Verifying..."):
        response = call_perplexity(f"Verify with current data: {fact_query}")
        if 'choices' in response:
            st.write("**Verification Result:**")
            st.write(response['choices'][0]['message']['content'])
        else:
            st.error(f"Verification failed: {response.get('error', 'Unknown')}")

# Topic Data Fetch
st.subheader("🧠 Fetch Topic Data")
topic = st.text_input("Enter topic:", placeholder="e.g., renewable energy")
if st.button("Fetch Data", disabled=not (topic and perplexity_key)):
    with st.spinner("Fetching data..."):
        response = call_perplexity(f"Latest data and statistics on {topic} 2024")
        if 'choices' in response:
            content = response['choices'][0]['message']['content']
            st.session_state.research_data.append({
                'topic': topic,
                'missing_info': 'General overview',
                'research_content': content,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
            })
            st.success("Data fetched!")
            st.rerun()
        else:
            st.error(f"Fetch failed: {response.get('error', 'Unknown')}")
