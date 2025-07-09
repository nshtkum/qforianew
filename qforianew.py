import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import re
import requests
from datetime import datetime
import time
from typing import List, Dict, Any
from urllib.parse import urlparse
import urllib.robotparser
from bs4 import BeautifulSoup
import hashlib

# App config
st.set_page_config(page_title="Qforia Pro Enhanced", layout="wide")

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .url-input-section {
        background: #e8f5e8;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .data-point {
        background: #e7f3ff;
        border-left: 4px solid #0066cc;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .enhancement-suggestion {
        background: #f0fff0;
        border-left: 4px solid #28a745;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .missing-topic {
        background: #fff5f5;
        border-left: 4px solid #dc3545;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .content-analysis {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 Qforia Pro Enhanced: URL Content Analysis Tool</h1>
    <p>AI-Powered Content Analysis, Missing Context Detection & Enhancement Suggestions</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    default_values = {
        'fanout_results': None,
        'research_data': [],
        'api_usage': {'gemini_calls': 0, 'perplexity_calls': 0},
        'content_analysis': None,
        'enhancement_suggestions': None,
        'scraped_content': None,
        'missing_context_data': [],
        'content_metadata': {}
    }
    
    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# Sidebar
st.sidebar.header("🔧 Configuration")

# API Keys with better error handling
def get_api_keys():
    """Get API keys from secrets or user input"""
    try:
        gemini_key = st.secrets["api_keys"]["GEMINI_API_KEY"]
        perplexity_key = st.secrets["api_keys"]["PERPLEXITY_API_KEY"]
        st.sidebar.success("🔑 API Keys loaded from secrets")
        return gemini_key, perplexity_key, True
    except Exception as e:
        st.sidebar.info("🔑 Enter API keys manually")
        gemini_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key")
        perplexity_key = st.sidebar.text_input("Perplexity API Key", type="password", help="Enter your Perplexity API key")
        
        # Validate keys
        keys_valid = True
        if not perplexity_key or not perplexity_key.startswith('pplx-'):
            st.sidebar.warning("⚠️ Valid Perplexity API key required (starts with 'pplx-')")
            keys_valid = False
        if not gemini_key:
            st.sidebar.warning("⚠️ Gemini API key required")
            keys_valid = False
            
        return gemini_key, perplexity_key, keys_valid

gemini_key, perplexity_key, api_keys_valid = get_api_keys()

# API Usage Display
st.sidebar.subheader("📊 API Usage")
st.sidebar.metric("Gemini Calls", st.session_state.api_usage['gemini_calls'])
st.sidebar.metric("Perplexity Calls", st.session_state.api_usage['perplexity_calls'])
estimated_cost = (st.session_state.api_usage['perplexity_calls'] * 0.002) + (st.session_state.api_usage['gemini_calls'] * 0.001)
st.sidebar.metric("Estimated Cost", f"${estimated_cost:.3f}")

# Configure Gemini
def setup_gemini(api_key):
    """Setup Gemini AI model"""
    if not api_key:
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        st.sidebar.success("✅ Gemini configured")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Gemini setup failed: {str(e)}")
        return None

model = setup_gemini(gemini_key)

# Enhanced URL scraping function
def scrape_url_content(url):
    """Scrape content from URL with robust error handling"""
    try:
        # Validate and normalize URL
        if not url.strip():
            return None, "Empty URL provided"
            
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url.strip()
        
        # Headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Make request with timeout
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()
        
        # Parse content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript']):
            element.decompose()
        
        # Try to find main content
        content_selectors = [
            'article', 'main', '[role="main"]', '.content', '.post-content', 
            '.entry-content', '.article-content', '.post-body', '.content-body',
            '.story-body', '.article-body', '.post', '.entry'
        ]
        
        main_content = None
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements and elements[0].get_text(strip=True):
                main_content = elements[0]
                break
        
        # Fallback to body if no main content found
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Extract text content
        text = main_content.get_text(separator='\n', strip=True)
        
        # Clean up text
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 10:  # Only keep substantial lines
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Basic validation
        if len(text.strip()) < 100:
            return None, "Content too short or unable to extract meaningful text"
        
        # Extract metadata
        title = soup.title.string if soup.title else 'No title found'
        title = title.strip() if title else 'No title found'
        
        # Try to extract description
        description = ""
        meta_desc = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
        if meta_desc and meta_desc.get('content'):
            description = meta_desc['content'].strip()
        
        metadata = {
            'title': title,
            'description': description,
            'word_count': len(text.split()),
            'char_count': len(text),
            'url': url,
            'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return {'content': text, 'metadata': metadata}, None
        
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {str(e)}"
    except Exception as e:
        return None, f"Parsing error: {str(e)}"

# API Functions
def call_perplexity_api(query):
    """Call Perplexity API for research"""
    if not perplexity_key or not perplexity_key.startswith('pplx-'):
        return {"error": "Invalid Perplexity API key"}
    
    try:
        headers = {
            "Authorization": f"Bearer {perplexity_key}",
            "Content-Type": "application/json"
        }
        
        # Updated model names as per Perplexity API documentation
        models_to_try = [
            "llama-3.1-sonar-huge-128k-online",
            "llama-3.1-sonar-large-128k-online", 
            "llama-3.1-70b-instruct",
            "mixtral-8x7b-instruct"
        ]
        
        for model_name in models_to_try:
            data = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a helpful research assistant. Provide detailed, factual information with specific numbers, statistics, and recent data where available. Include sources when possible."
                    },
                    {"role": "user", "content": query}
                ],
                "temperature": 0.2,
                "max_tokens": 1000
            }
            
            try:
                response = requests.post(
                    "https://api.perplexity.ai/chat/completions", 
                    headers=headers, 
                    json=data, 
                    timeout=30
                )
                
                if response.status_code == 200:
                    st.session_state.api_usage['perplexity_calls'] += 1
                    return response.json()
                elif response.status_code == 400:
                    # Model not available, try next one
                    continue
                else:
                    # Other error, return immediately
                    return {"error": f"API call failed with status {response.status_code}: {response.text}"}
                    
            except requests.exceptions.RequestException as e:
                if model_name == models_to_try[-1]:  # Last model
                    return {"error": f"Request failed: {str(e)}"}
                continue
        
        return {"error": "All models failed or are unavailable"}
    
    except Exception as e:
        return {"error": f"Exception occurred: {str(e)}"}

def extract_data_points(text):
    """Extract numerical data points and key facts from text"""
    if not text or len(text.strip()) < 10:
        return []
    
    data_points = []
    
    # Split into sentences for better context
    sentences = re.split(r'[.!?]+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:  # Skip very short sentences
            continue
            
        # Patterns for different types of data
        patterns = [
            (r'(\d+(?:\.\d+)?%)', 'Percentage'),
            (r'(\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:billion|million|thousand|crore|lakh|B|M|K))?)', 'Currency'),
            (r'(₹\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:billion|million|thousand|crore|lakh))?)', 'Currency'),
            (r'(\d+(?:,\d{3})*(?:\.\d+)?\s*(?:billion|million|thousand|crore|lakh))', 'Large Number'),
            (r'(\d{4}(?:-\d{4})?)', 'Year'),
            (r'(\d+(?:\.\d+)?\s*(?:sq\s*ft|acres|hectares|sqft|km²|mi²))', 'Area'),
            (r'(\d+(?:\.\d+)?\s*(?:years?|months?|days?|hrs?|hours?))', 'Time Period'),
            (r'(\d+(?:\.\d+)?\s*(?:tons?|kg|pounds?|lbs))', 'Weight'),
            (r'(\d+(?:\.\d+)?\s*(?:°C|°F|degrees?))', 'Temperature'),
        ]
        
        for pattern, data_type in patterns:
            try:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    # Clean up the context
                    context = sentence.replace('\n', ' ').strip()
                    if len(context) > 200:
                        # Truncate long contexts
                        match_pos = context.lower().find(match.lower())
                        if match_pos != -1:
                            start = max(0, match_pos - 100)
                            end = min(len(context), match_pos + 100)
                            context = context[start:end]
                            if start > 0:
                                context = "..." + context
                            if end < len(sentence):
                                context = context + "..."
                    
                    data_points.append({
                        'value': match,
                        'type': data_type,
                        'context': context
                    })
            except Exception:
                continue  # Skip problematic patterns
    
    # Remove duplicates
    seen = set()
    unique_data_points = []
    for dp in data_points:
        try:
            identifier = (dp['value'].lower(), dp['type'])
            if identifier not in seen:
                seen.add(identifier)
                unique_data_points.append(dp)
        except Exception:
            continue
    
    return unique_data_points

def identify_missing_context(content, primary_keyword=""):
    """Identify missing context using Gemini AI"""
    if not model:
        return None, "Gemini model not configured"
    
    if not content or len(content.strip()) < 50:
        return None, "Content too short for analysis"
    
    try:
        # Truncate content if too long
        content_for_analysis = content[:8000] if len(content) > 8000 else content
        
        prompt = f"""
        Analyze this article content and identify what's missing for a comprehensive understanding. 
        Focus on identifying gaps in information, missing data points, and areas that need more context.
        Primary keyword/topic: {primary_keyword}

        Content to analyze:
        {content_for_analysis}

        Provide analysis in this exact JSON format (ensure valid JSON):
        {{
            "main_topics": ["topic1", "topic2", "topic3"],
            "content_type": "blog_post",
            "target_audience": "general",
            "key_strengths": ["strength1", "strength2"],
            "missing_context": [
                {{
                    "category": "Statistics",
                    "missing_info": "What specific data is missing",
                    "importance": "high",
                    "reason": "Why this information is important"
                }}
            ],
            "data_gaps": [
                {{
                    "type": "financial",
                    "description": "What data is missing",
                    "research_query": "Suggested search query to find this data"
                }}
            ],
            "enhancement_opportunities": [
                {{
                    "area": "Area to enhance",
                    "suggestion": "Specific enhancement suggestion",
                    "priority": "high"
                }}
            ],
            "fact_check_needed": [
                "Statement or claim that should be verified"
            ]
        }}
        """
        
        response = model.generate_content(prompt)
        st.session_state.api_usage['gemini_calls'] += 1
        
        # Clean and parse JSON response
        json_text = response.text.strip()
        
        # Remove markdown formatting if present
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        
        json_text = json_text.strip()
        
        # Parse JSON
        analysis = json.loads(json_text)
        return analysis, None
        
    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {str(e)}"
    except Exception as e:
        return None, f"Error analyzing content: {str(e)}"

def research_missing_context(missing_context_items):
    """Use Perplexity to research missing context and data gaps"""
    if not missing_context_items:
        return []
    
    research_results = []
    
    for item in missing_context_items[:10]:  # Limit to 10 items to control costs
        try:
            # Create research query
            if 'research_query' in item and item['research_query']:
                query = item['research_query']
            else:
                # Create a research query from the missing info
                category = item.get('category', '')
                missing_info = item.get('missing_info', '') or item.get('description', '')
                query = f"Latest data and statistics about {missing_info} {category} 2024"
            
            # Research with Perplexity
            research_response = call_perplexity_api(query)
            
            if 'choices' in research_response and research_response['choices']:
                content = research_response['choices'][0]['message']['content']
                data_points = extract_data_points(content)
                
                research_results.append({
                    'original_query': query,
                    'category': item.get('category', 'Unknown'),
                    'missing_info': item.get('missing_info', '') or item.get('description', ''),
                    'research_content': content,
                    'data_points': data_points,
                    'importance': item.get('importance', 'medium'),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
                })
            else:
                error_msg = research_response.get('error', 'Unknown error')
                research_results.append({
                    'original_query': query,
                    'category': item.get('category', 'Unknown'),
                    'missing_info': item.get('missing_info', '') or item.get('description', ''),
                    'research_content': f"Error: {error_msg}",
                    'data_points': [],
                    'importance': item.get('importance', 'medium'),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
                })
            
            # Add delay to respect rate limits
            time.sleep(1)
            
        except Exception as e:
            research_results.append({
                'original_query': str(item),
                'category': 'Unknown',
                'missing_info': 'Processing error',
                'research_content': f"Error processing item: {str(e)}",
                'data_points': [],
                'importance': 'low',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
            })
    
    return research_results

# ========================================
# MAIN FEATURE: URL CONTENT ANALYSIS
# ========================================

st.markdown("""
<div class="url-input-section">
    <h2>🔗 URL Content Analysis & Enhancement</h2>
    <p><strong>Analyze any article URL, identify missing context, and get AI-powered data to fill the gaps!</strong></p>
</div>
""", unsafe_allow_html=True)

# URL Input Section
st.subheader("🌐 Analyze Article from URL")

url_input = st.text_input(
    "📎 Enter Article URL to Analyze:",
    placeholder="https://example.com/article-to-analyze",
    help="Paste the full URL of any article you want to analyze for missing context"
)

primary_keyword = st.text_input(
    "🎯 Primary Keyword/Topic (Optional):",
    placeholder="e.g., 'real estate', 'investment', 'technology'",
    help="Enter the main topic to focus the analysis"
)

col1, col2 = st.columns(2)

with col1:
    analyze_button = st.button("🔍 Analyze URL Content", type="primary", disabled=not api_keys_valid)

with col2:
    manual_input_button = st.button("📝 Use Manual Input", help="Paste content manually if URL scraping fails")

if analyze_button and url_input and api_keys_valid:
    with st.spinner("🔄 Processing URL..."):
        
        # Step 1: Scrape URL
        st.info("Step 1: Scraping URL content...")
        scraped_data, error = scrape_url_content(url_input)
        
        if scraped_data:
            content = scraped_data['content']
            metadata = scraped_data['metadata']
            
            # Store in session state
            st.session_state.scraped_content = content
            st.session_state.content_metadata = metadata
            
            # Display scraping success
            st.success("✅ Content successfully scraped!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Word Count", metadata['word_count'])
            with col2:
                st.metric("Characters", metadata['char_count'])
            with col3:
                st.metric("Title", "✅" if metadata['title'] != 'No title found' else "❌")
            with col4:
                st.metric("Description", "✅" if metadata.get('description') else "❌")
            
            # Show content preview
            with st.expander("👀 Scraped Content Preview"):
                st.markdown(f"**Title:** {metadata['title']}")
                st.markdown(f"**Description:** {metadata.get('description', 'N/A')}")
                st.markdown(f"**Source:** {metadata['url']}")
                preview_text = content[:1500] + "..." if len(content) > 1500 else content
                st.text_area("Content Preview", preview_text, height=200, disabled=True)
            
            # Step 2: Quick data extraction
            st.info("Step 2: Extracting data points...")
            data_points = extract_data_points(content)
            
            if data_points:
                st.subheader("📊 Initial Data Points Found")
                for i, dp in enumerate(data_points[:8]):  # Show first 8
                    st.markdown(f"""
                    <div class="data-point">
                        <strong>{dp['value']}</strong> ({dp['type']})<br>
                        <small>{dp['context'][:150]}...</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                if len(data_points) > 8:
                    st.info(f"... and {len(data_points) - 8} more data points found")
            else:
                st.info("No specific data points found in the content")
            
            # Step 3: AI Analysis with Gemini
            st.info("Step 3: Analyzing content with Gemini AI...")
            analysis, analysis_error = identify_missing_context(content, primary_keyword)
            
            if analysis:
                st.session_state.content_analysis = analysis
                st.success("✅ AI analysis completed!")
                
                # Display analysis results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📝 Main Topics**")
                    for topic in analysis.get('main_topics', [])[:5]:
                        st.markdown(f"• {topic}")
                    
                    st.markdown("**💪 Content Strengths**")
                    for strength in analysis.get('key_strengths', [])[:3]:
                        st.markdown(f"✅ {strength}")
                
                with col2:
                    st.markdown(f"**📋 Content Type:** {analysis.get('content_type', 'Unknown')}")
                    st.markdown(f"**🎯 Target Audience:** {analysis.get('target_audience', 'Unknown')}")
                    
                    missing_count = len(analysis.get('missing_context', []))
                    data_gaps_count = len(analysis.get('data_gaps', []))
                    st.metric("Missing Context Items", missing_count)
                    st.metric("Data Gaps Identified", data_gaps_count)
                
                # Step 4: Research missing context with Perplexity
                missing_context = analysis.get('missing_context', []) + analysis.get('data_gaps', [])
                
                if missing_context:
                    st.info("Step 4: Researching missing context with Perplexity...")
                    
                    # Show what will be researched
                    st.subheader("🔍 Context Gaps Identified")
                    with st.expander("👀 What will be researched"):
                        for item in missing_context[:5]:  # Show first 5
                            importance = item.get('importance', 'medium')
                            emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(importance, "🟡")
                            missing_info = item.get('missing_info', '') or item.get('description', '')
                            st.markdown(f"{emoji} **{item.get('category', 'Unknown')}:** {missing_info}")
                    
                    # Research with progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    research_results = research_missing_context(missing_context)
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Research completed!")
                    
                    # Store research results
                    st.session_state.missing_context_data = research_results
                    
                    # Clear progress indicators
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success("🎉 Complete analysis finished! Scroll down to see results.")
                    st.rerun()
                
                else:
                    st.success("✅ No significant context gaps identified. The content appears comprehensive!")
            
            else:
                st.error(f"❌ AI analysis failed: {analysis_error}")
        
        else:
            st.error(f"❌ Failed to scrape content: {error}")
            st.info("💡 Please try the manual input option below")

# Manual Content Input
if manual_input_button:
    st.markdown("---")
    st.subheader("📝 Manual Content Input")
    
    manual_content = st.text_area(
        "Paste article content here:",
        height=300,
        placeholder="Paste the complete article content here for analysis..."
    )
    
    if st.button("📊 Analyze Manual Content", type="secondary", disabled=not api_keys_valid) and manual_content:
        # Store manual content
        st.session_state.scraped_content = manual_content
        st.session_state.content_metadata = {
            'title': 'Manual Input',
            'description': '',
            'word_count': len(manual_content.split()),
            'char_count': len(manual_content),
            'url': 'Manual Input',
            'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        st.success("✅ Manual content loaded. Starting analysis...")
        
        # Proceed with analysis
        with st.spinner("Analyzing content..."):
            analysis, analysis_error = identify_missing_context(manual_content, primary_keyword)
            
            if analysis:
                st.session_state.content_analysis = analysis
                
                # Research missing context
                missing_context = analysis.get('missing_context', []) + analysis.get('data_gaps', [])
                if missing_context:
                    research_results = research_missing_context(missing_context)
                    st.session_state.missing_context_data = research_results
                
                st.success("✅ Analysis completed!")
                st.rerun()
            else:
                st.error(f"❌ Analysis failed: {analysis_error}")

# ========================================
# DISPLAY ANALYSIS RESULTS
# ========================================

# Display stored analysis results
if st.session_state.content_analysis:
    st.markdown("---")
    st.header("📊 Content Analysis Results")
    
    analysis = st.session_state.content_analysis
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Analysis Status", "✅ Complete")
    with col2:
        st.metric("Content Type", analysis.get('content_type', 'Unknown'))
    with col3:
        st.metric("Target Audience", analysis.get('target_audience', 'Unknown'))
    with col4:
        missing_items = len(analysis.get('missing_context', [])) + len(analysis.get('data_gaps', []))
        st.metric("Context Gaps", missing_items)
    
    # Main analysis display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Content Overview")
        
        st.markdown("**Main Topics:**")
        for topic in analysis.get('main_topics', []):
            st.markdown(f"• {topic}")
        
        st.markdown("**Content Strengths:**")
        for strength in analysis.get('key_strengths', []):
            st.markdown(f"✅ {strength}")
    
    with col2:
        st.subheader("🔍 Missing Context Analysis")
        
        missing_context = analysis.get('missing_context', [])
        if missing_context:
            for item in missing_context[:3]:
                importance = item.get('importance', 'medium')
                emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(importance, "🟡")
                st.markdown(f"""
                <div class="missing-topic">
                    {emoji} <strong>{item.get('category', 'Unknown')}</strong><br>
                    {item.get('missing_info', 'No description')}
                </div>
                """, unsafe_allow_html=True)
        
        data_gaps = analysis.get('data_gaps', [])
        if data_gaps:
            st.markdown("**Data Gaps Identified:**")
            for gap in data_gaps[:3]:
                st.markdown(f"• {gap.get('description', 'No description')}")

# Display research results if available
if st.session_state.missing_context_data:
    st.markdown("---")
    st.header("🔬 Missing Context Research Results")
    
    research_results = st.session_state.missing_context_data
    
    # Summary metrics
    total_data_points = sum(len(r['data_points']) for r in research_results)
    high_priority_items = sum(1 for r in research_results if r.get('importance') == 'high')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Areas Researched", len(research_results))
    with col2:
        st.metric("New Data Found", total_data_points)
    with col3:
        st.metric("High Priority", high_priority_items)
    with col4:
        successful = sum(1 for r in research_results if 'Error:' not in r['research_content'])
        st.metric("Success Rate", f"{(successful/len(research_results)*100):.0f}%")
    
    # Filter and display options
    filter_importance = st.selectbox(
        "Filter by importance:",
        ["All", "High Priority", "Medium Priority", "Low Priority"]
    )
    
    filtered_results = research_results
    if filter_importance != "All":
        importance_map = {"High Priority": "high", "Medium Priority": "medium", "Low Priority": "low"}
        filtered_results = [r for r in research_results if r.get('importance') == importance_map[filter_importance]]
    
    # Display research results
    for i, result in enumerate(filtered_results):
        importance = result.get('importance', 'medium')
        emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(importance, "🟡")
        
        with st.expander(f"{emoji} {result['category']}: {result['missing_info'][:80]}..."):
            
            # Research query used
            st.markdown("**🔍 Research Query:**")
            st.code(result['original_query'])
            
            # Research findings
            st.markdown("**📋 Research Findings:**")
            if 'Error:' in result['research_content']:
                st.error(result['research_content'])
            else:
                st.markdown(result['research_content'])
                
                # Display data points found
                if result['data_points']:
                    st.markdown("**📊 Key Data Points Found:**")
                    
                    # Create a nice table for data points
                    data_for_table = []
                    for dp in result['data_points']:
                        data_for_table.append({
                            'Value': dp['value'],
                            'Type': dp['type'],
                            'Context': dp['context'][:100] + "..." if len(dp['context']) > 100 else dp['context']
                        })
                    
                    if data_for_table:
                        df = pd.DataFrame(data_for_table)
                        st.dataframe(df, hide_index=True, use_container_width=True)
                else:
                    st.info("No specific data points extracted from this research")

# Export functionality
if st.session_state.content_analysis or st.session_state.missing_context_data:
    
    st.markdown("---")
    st.header("📤 Export Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Summary report
        if st.button("📋 Generate Summary Report"):
            summary_report = f"""
QFORIA PRO ENHANCED - CONTENT ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source: {st.session_state.content_metadata.get('url', 'Manual Input')}

CONTENT OVERVIEW:
- Title: {st.session_state.content_metadata.get('title', 'N/A')}
- Word Count: {st.session_state.content_metadata.get('word_count', 'N/A')}
- Content Type: {st.session_state.content_analysis.get('content_type', 'N/A') if st.session_state.content_analysis else 'N/A'}
- Target Audience: {st.session_state.content_analysis.get('target_audience', 'N/A') if st.session_state.content_analysis else 'N/A'}

MAIN TOPICS:
"""
            if st.session_state.content_analysis:
                for topic in st.session_state.content_analysis.get('main_topics', []):
                    summary_report += f"- {topic}\n"
                
                summary_report += "\nMISSING CONTEXT IDENTIFIED:\n"
                for item in st.session_state.content_analysis.get('missing_context', []):
                    summary_report += f"- {item.get('category', 'Unknown')}: {item.get('missing_info', 'N/A')}\n"
            
            if st.session_state.missing_context_data:
                summary_report += f"\nRESEARCH COMPLETED:\n"
                summary_report += f"- Areas researched: {len(st.session_state.missing_context_data)}\n"
                total_data_points = sum(len(r['data_points']) for r in st.session_state.missing_context_data)
                summary_report += f"- New data points found: {total_data_points}\n"
            
            st.download_button(
                "📥 Download Summary",
                data=summary_report,
                file_name=f"qforia_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )
    
    with col2:
        # Full JSON export
        if st.button("📊 Export Full Data"):
            export_data = {
                'metadata': st.session_state.content_metadata,
                'analysis': st.session_state.content_analysis,
                'research_results': st.session_state.missing_context_data,
                'api_usage': st.session_state.api_usage,
                'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            json_data = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                "📥 Download JSON",
                data=json_data,
                file_name=f"qforia_full_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
    
    with col3:
        # Data points CSV
        if st.button("📈 Export Data Points"):
            all_data_points = []
            
            # Original content data points
            if st.session_state.scraped_content:
                original_points = extract_data_points(st.session_state.scraped_content)
                for dp in original_points:
                    all_data_points.append({
                        'Source': 'Original Content',
                        'Category': 'Found in Content',
                        'Value': dp['value'],
                        'Type': dp['type'],
                        'Context': dp['context']
                    })
            
            # Research data points
            for result in st.session_state.missing_context_data:
                for dp in result['data_points']:
                    all_data_points.append({
                        'Source': 'Research',
                        'Category': result['category'],
                        'Value': dp['value'],
                        'Type': dp['type'],
                        'Context': dp['context']
                    })
            
            if all_data_points:
                df = pd.DataFrame(all_data_points)
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    data=csv_data,
                    file_name=f"qforia_data_points_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )

# ========================================
# ADDITIONAL FEATURES
# ========================================

st.markdown("---")
st.header("🔧 Additional Tools")

tool_selection = st.selectbox(
    "Choose additional tool:",
    ["Select a tool...", "🔍 Quick Fact Checker", "📊 Data Point Extractor", "🧠 Topic Analyzer"]
)

if tool_selection == "🔍 Quick Fact Checker":
    st.subheader("🔍 Quick Fact Checker")
    
    fact_query = st.text_input(
        "Enter a claim or topic to verify:",
        placeholder="e.g., 'Global smartphone market grew by 12% in 2024'"
    )
    
    if st.button("✅ Verify Fact") and fact_query and api_keys_valid:
        with st.spinner("Fact-checking with Perplexity..."):
            verification_query = f"Verify this claim with current data and sources: {fact_query}"
            response = call_perplexity_api(verification_query)
            
            if 'choices' in response:
                verification_result = response['choices'][0]['message']['content']
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**📋 Verification Result:**")
                    st.markdown(verification_result)
                
                with col2:
                    # Extract any data points from verification
                    data_points = extract_data_points(verification_result)
                    st.metric("Data Points Found", len(data_points))
                    
                    if data_points:
                        st.markdown("**Key Data:**")
                        for dp in data_points[:3]:
                            st.markdown(f"• {dp['value']} ({dp['type']})")
            else:
                st.error(f"Verification failed: {response.get('error', 'Unknown error')}")

elif tool_selection == "📊 Data Point Extractor":
    st.subheader("📊 Data Point Extractor")
    
    text_to_analyze = st.text_area(
        "Paste text to extract data points:",
        height=200,
        placeholder="Paste any text content here to extract numerical data, statistics, and key facts..."
    )
    
    if st.button("🔍 Extract Data Points") and text_to_analyze:
        data_points = extract_data_points(text_to_analyze)
        
        if data_points:
            st.success(f"✅ Found {len(data_points)} data points!")
            
            # Create DataFrame for better display
            df_data = []
            for dp in data_points:
                df_data.append({
                    'Value': dp['value'],
                    'Type': dp['type'],
                    'Context': dp['context'][:150] + "..." if len(dp['context']) > 150 else dp['context']
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, hide_index=True, use_container_width=True)
            
            # Download option
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📥 Download Data Points",
                data=csv_data,
                file_name=f"extracted_data_points_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No data points found in the provided text")

elif tool_selection == "🧠 Topic Analyzer":
    st.subheader("🧠 Topic Analyzer")
    
    topic_text = st.text_area(
        "Enter content to analyze topics:",
        height=200,
        placeholder="Paste content here to analyze main topics and themes..."
    )
    
    if st.button("🔍 Analyze Topics") and topic_text and api_keys_valid and model:
        with st.spinner("Analyzing topics with Gemini..."):
            topic_prompt = f"""
            Analyze this content and extract the main topics, themes, and key concepts.
            
            Content: {topic_text[:4000]}
            
            Provide analysis in JSON format:
            {{
                "main_topics": ["topic1", "topic2"],
                "themes": ["theme1", "theme2"],
                "key_concepts": ["concept1", "concept2"],
                "sentiment": "positive",
                "complexity_level": "intermediate",
                "summary": "Brief summary of the content"
            }}
            """
            
            try:
                response = model.generate_content(topic_prompt)
                st.session_state.api_usage['gemini_calls'] += 1
                
                json_text = response.text.strip()
                if json_text.startswith("```json"):
                    json_text = json_text[7:]
                if json_text.endswith("```"):
                    json_text = json_text[:-3]
                
                topic_analysis = json.loads(json_text.strip())
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📝 Main Topics:**")
                    for topic in topic_analysis.get('main_topics', []):
                        st.markdown(f"• {topic}")
                    
                    st.markdown("**🎨 Themes:**")
                    for theme in topic_analysis.get('themes', []):
                        st.markdown(f"• {theme}")
                
                with col2:
                    st.markdown("**🧠 Key Concepts:**")
                    for concept in topic_analysis.get('key_concepts', []):
                        st.markdown(f"• {concept}")
                    
                    st.markdown(f"**📊 Sentiment:** {topic_analysis.get('sentiment', 'Unknown')}")
                    st.markdown(f"**📈 Complexity:** {topic_analysis.get('complexity_level', 'Unknown')}")
                
                st.markdown("**📋 Summary:**")
                st.markdown(topic_analysis.get('summary', 'No summary available'))
                
            except Exception as e:
                st.error(f"Topic analysis failed: {e}")

# ========================================
# FOOTER & UTILITIES
# ========================================

st.markdown("---")
st.subheader("🎯 Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🗑️ Clear All Data"):
        # Clear session state
        keys_to_clear = ['scraped_content', 'content_metadata', 'content_analysis', 
                        'missing_context_data', 'fanout_results', 'research_data']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.success("✅ All data cleared!")
        st.rerun()

with col2:
    if st.button("💡 Usage Guide"):
        st.info("""
        **How to use Qforia Pro Enhanced:**
        
        1. **URL Analysis**: Enter any article URL to automatically scrape and analyze content
        2. **AI Analysis**: Gemini identifies missing context and data gaps
        3. **Auto Research**: Perplexity fills in the missing information
        4. **Export Results**: Download analysis, data points, and research findings
        
        **Pro Tips:**
        - Use manual input if URL scraping fails
        - Add primary keyword for focused analysis
        - Check API usage to monitor costs
        - Export data for offline analysis
        """)

with col3:
    if st.button("🔧 API Status"):
        gemini_status = "✅ Connected" if model else "❌ Not configured"
        perplexity_status = "✅ Valid" if perplexity_key and perplexity_key.startswith('pplx-') else "❌ Invalid"
        
        st.info(f"""
        **API Configuration Status:**
        
        🤖 **Gemini AI**: {gemini_status}
        🔍 **Perplexity**: {perplexity_status}
        
        **Current Usage:**
        - Gemini calls: {st.session_state.api_usage['gemini_calls']}
        - Perplexity calls: {st.session_state.api_usage['perplexity_calls']}
        - Estimated cost: ${estimated_cost:.3f}
        """)

with col4:
    if st.button("📊 Session Summary"):
        features_used = []
        if st.session_state.scraped_content:
            features_used.append("Content Analysis")
        if st.session_state.missing_context_data:
            features_used.append("Missing Context Research")
        if st.session_state.content_analysis:
            features_used.append("AI Analysis")
        
        data_points_found = 0
        if st.session_state.scraped_content:
            data_points_found += len(extract_data_points(st.session_state.scraped_content))
        if st.session_state.missing_context_data:
            data_points_found += sum(len(r['data_points']) for r in st.session_state.missing_context_data)
        
        st.info(f"""
        **Session Summary:**
        
        📈 **Features Used**: {len(features_used)}
        📊 **Total Data Points**: {data_points_found}
        🔄 **API Calls**: {st.session_state.api_usage['gemini_calls'] + st.session_state.api_usage['perplexity_calls']}
        💰 **Cost**: ${estimated_cost:.3f}
        
        **Active Features**: {', '.join(features_used) if features_used else 'None'}
        """)

st.markdown("---")
st.markdown("**🚀 Qforia Pro Enhanced v4.0** - Advanced URL Content Analysis & Missing Context Research Tool")
st.markdown("*Powered by Gemini AI for content analysis and Perplexity for real-time research*")
