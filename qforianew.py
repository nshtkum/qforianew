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
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer']):
            element.decompose()
        
        # Extract content
        content = soup.find('article') or soup.find('main') or soup.find('body')
        text = content.get_text(separator=' ', strip=True) if content else ""
        
        if len(text) < 100:
            return None, "Content too short"
        
        title = soup.title.string.strip() if soup.title and soup.title.string else "No title"
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
        "model": "llama-3.1-sonar-small-128k-online",  # Updated to current model
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

        Return ONLY valid JSON with this structure:
        {{
            "main_topics": ["topic1", "topic2"],
            "missing_context": [
                {{"topic": "topic", "missing_info": "description", "research_query": "query"}}
            ]
        }}
        """
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        
        # Clean up JSON response
        if json_text.startswith("```json"):
            json_text = json_text[7:-3]
        elif json_text.startswith("```"):
            json_text = json_text[3:-3]
        
        # Parse JSON
        parsed_json = json.loads(json_text)
        
        # Validate structure
        if not isinstance(parsed_json, dict):
            raise ValueError("Response is not a valid JSON object")
        
        if 'main_topics' not in parsed_json:
            parsed_json['main_topics'] = []
        if 'missing_context' not in parsed_json:
            parsed_json['missing_context'] = []
            
        return parsed_json, None
    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {e}"
    except Exception as e:
        return None, f"Analysis error: {e}"

# Research missing context
def research_missing_context(missing_items):
    results = []
    for item in missing_items[:5]:
        query = item.get('research_query', f"Latest data on {item.get('topic', 'unknown')} {item.get('missing_info', '')}")
        time.sleep(1)  # Delay to avoid rate limits
        response = call_perplexity(query)
        
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            results.append({
                'topic': item.get('topic', 'Unknown'),
                'missing_info': item.get('missing_info', 'Unknown'),
                'research_content': content,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
            })
        else:
            results.append({
                'topic': item.get('topic', 'Unknown'),
                'missing_info': item.get('missing_info', 'Unknown'),
                'research_content': f"Error: {response.get('error', 'Unknown error')}",
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
                    with st.spinner("Researching missing context..."):
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
    df_data = []
    for r in st.session_state.research_data:
        research_content = r.get('research_content', '')
        if isinstance(research_content, str) and len(research_content) > 200:
            research_content = research_content[:200] + "..."
        
        df_data.append({
            'Topic': r.get('topic', 'Unknown'),
            'Missing Info': r.get('missing_info', 'Unknown'),
            'Research Findings': research_content,
            'Timestamp': r.get('timestamp', 'Unknown')
        })
    
    if df_data:
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)

# Fact Checker
st.subheader("🔍 Fact Checker")
fact_query = st.text_input("Enter claim to verify:", placeholder="e.g., Smartphone market grew 12% in 2024")

if st.button("Verify Fact", disabled=not (fact_query and perplexity_key)):
    with st.spinner("Verifying..."):
        response = call_perplexity(f"Verify with current data: {fact_query}")
        if 'choices' in response and len(response['choices']) > 0:
            st.write("**Verification Result:**")
            st.write(response['choices'][0]['message']['content'])
        else:
            st.error(f"Verification failed: {response.get('error', 'Unknown error')}")

# Topic Data Fetch
st.subheader("🧠 Fetch Topic Data")
topic = st.text_input("Enter topic:", placeholder="e.g., renewable energy")

if st.button("Fetch Data", disabled=not (topic and perplexity_key)):
    with st.spinner("Fetching data..."):
        response = call_perplexity(f"Latest data and statistics on {topic} 2024")
        if 'choices' in response and len(response['choices']) > 0:
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
            st.error(f"Fetch failed: {response.get('error', 'Unknown error')}")

# Clear results button
if st.button("Clear All Results"):
    st.session_state.analysis_results = None
    st.session_state.research_data = []
    st.rerun()

# Footer
st.markdown("---")
st.write("**Qforia Simplified** - URL Analysis & Fact Checking Tool")
st.write("*Powered by Gemini AI and Perplexity*")
