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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 Qforia Pro Enhanced: Query Fan-Out & Content Analysis Tool</h1>
    <p>AI-Powered Query Expansion, Real-Time Fact Verification & Content Enhancement</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'fanout_results' not in st.session_state:
    st.session_state.fanout_results = None
if 'research_data' not in st.session_state:
    st.session_state.research_data = []
if 'api_usage' not in st.session_state:
    st.session_state.api_usage = {'gemini_calls': 0, 'perplexity_calls': 0}
if 'content_analysis' not in st.session_state:
    st.session_state.content_analysis = None
if 'enhancement_suggestions' not in st.session_state:
    st.session_state.enhancement_suggestions = None

# Sidebar
st.sidebar.header("🔧 Configuration")

# API Keys
try:
    gemini_key = st.secrets["api_keys"]["GEMINI_API_KEY"]
    perplexity_key = st.secrets["api_keys"]["PERPLEXITY_API_KEY"]
    st.sidebar.success("🔑 API Keys loaded from secrets")
except:
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key")
    perplexity_key = st.sidebar.text_input("Perplexity API Key", type="password", help="Enter your Perplexity API key")

# Validate API keys
if not perplexity_key or not perplexity_key.startswith('pplx-'):
    st.sidebar.warning("⚠️ Valid Perplexity API key required (starts with 'pplx-')")
if not gemini_key:
    st.sidebar.warning("⚠️ Gemini API key required")

# API Usage Display
st.sidebar.subheader("📊 API Usage")
st.sidebar.metric("Gemini Calls", st.session_state.api_usage['gemini_calls'])
st.sidebar.metric("Perplexity Calls", st.session_state.api_usage['perplexity_calls'])
estimated_cost = (st.session_state.api_usage['perplexity_calls'] * 0.002) + (st.session_state.api_usage['gemini_calls'] * 0.001)
st.sidebar.metric("Estimated Cost", f"${estimated_cost:.3f}")

# Configure Gemini
if gemini_key:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Enhanced URL scraping function with fallback
def scrape_url_content(url, use_fallback=False):
    """Scrape content from URL with multiple fallback methods"""
    if use_fallback:
        return None, "Please paste content manually"
    
    try:
        # Check robots.txt first
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Try to respect robots.txt
        try:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{base_url}/robots.txt")
            rp.read()
            if not rp.can_fetch('*', url):
                return None, "Access blocked by robots.txt"
        except:
            pass  # Continue if robots.txt check fails
        
        # Multiple request strategies
        strategies = [
            # Strategy 1: Standard request
            lambda: requests.get(url, headers=headers, timeout=15),
            # Strategy 2: Session with cookies
            lambda: requests.Session().get(url, headers=headers, timeout=15),
            # Strategy 3: Different user agent
            lambda: requests.get(url, headers={**headers, 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}, timeout=15)
        ]
        
        response = None
        for i, strategy in enumerate(strategies):
            try:
                response = strategy()
                if response.status_code == 200:
                    break
                time.sleep(1)  # Brief delay between attempts
            except Exception as e:
                if i == len(strategies) - 1:  # Last strategy
                    return None, f"All request strategies failed: {str(e)}"
                continue
        
        if not response or response.status_code != 200:
            return None, f"HTTP Error: {response.status_code if response else 'No response'}"
        
        # Parse content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Try to extract main content
        content_selectors = [
            'article', 'main', '[role="main"]', '.content', '.post-content', 
            '.entry-content', '.article-content', '.post-body', '.content-body'
        ]
        
        main_content = None
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                main_content = elements[0]
                break
        
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Extract text
        text = main_content.get_text(separator=' ', strip=True)
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
        text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines
        
        # Extract metadata
        metadata = {
            'title': soup.title.string if soup.title else 'No title found',
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
    try:
        headers = {
            "Authorization": f"Bearer {perplexity_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "sonar-pro",
            "messages": [
                {"role": "system", "content": "You are a helpful research assistant. Provide detailed, factual information with specific numbers and statistics where available."},
                {"role": "user", "content": query}
            ],
            "temperature": 0.2,
            "max_tokens": 1000
        }
        
        response = requests.post("https://api.perplexity.ai/chat/completions", 
                               headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            st.session_state.api_usage['perplexity_calls'] += 1
            return response.json()
        elif response.status_code == 400:
            alternative_models = [
                "llama-3.1-sonar-large-128k-online",
                "llama-3.1-sonar-small-128k-online"
            ]
            
            for model in alternative_models:
                simple_data = {
                    "model": model,
                    "messages": [{"role": "user", "content": query}],
                    "max_tokens": 1000,
                    "temperature": 0.2
                }
                
                simple_response = requests.post("https://api.perplexity.ai/chat/completions", 
                                              headers=headers, json=simple_data, timeout=30)
                
                if simple_response.status_code == 200:
                    st.session_state.api_usage['perplexity_calls'] += 1
                    return simple_response.json()
            
            error_details = response.text if response.text else "No error details"
            return {"error": f"API call failed with status {response.status_code}. Details: {error_details}"}
        else:
            error_details = response.text if response.text else "No error details"
            return {"error": f"API call failed with status {response.status_code}. Details: {error_details}"}
    
    except Exception as e:
        return {"error": f"Exception occurred: {str(e)}"}

def extract_data_points(text):
    """Extract numerical data points from text"""
    data_points = []
    
    sentences = re.split(r'[.!?]+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
            
        patterns = [
            (r'(\d+(?:\.\d+)?%)', 'Percentage'),
            (r'(\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:billion|million|thousand|crore|lakh))?)', 'Currency'),
            (r'(₹\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:billion|million|thousand|crore|lakh))?)', 'Currency'),
            (r'(\d+(?:,\d{3})*(?:\.\d+)?\s*(?:billion|million|thousand|crore|lakh))', 'Large Number'),
            (r'(\d{4}(?:-\d{4})?)', 'Year'),
            (r'(\d+(?:\.\d+)?\s*(?:sq\s*ft|acres|hectares|sqft))', 'Area'),
            (r'(\d+(?:\.\d+)?\s*(?:years?|months?|days?))', 'Time Period'),
        ]
        
        for pattern, data_type in patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                
                context = sentence.replace('\n', ' ').strip()
                if len(context) > 150:
                    match_pos = context.find(match)
                    start = max(0, match_pos - 75)
                    end = min(len(context), match_pos + 75)
                    context = context[start:end]
                    if start > 0:
                        context = "..." + context
                    if end < len(sentence):
                        context = context + "..."
                
                data_points.append({
                    'value': match,
                    'type': data_type,
                    'description': context
                })
    
    seen = set()
    unique_data_points = []
    for dp in data_points:
        identifier = (dp['value'], dp['type'])
        if identifier not in seen:
            seen.add(identifier)
            unique_data_points.append(dp)
    
    return unique_data_points

def analyze_content_structure(content):
    """Analyze content structure and extract key topics"""
    try:
        prompt = f"""
        Analyze this article content and provide a detailed structural analysis:

        Content: {content[:4000]}...

        Provide analysis in this JSON format:
        {{
            "main_topics": ["topic1", "topic2", "topic3"],
            "content_type": "blog_post|news_article|guide|analysis",
            "tone": "professional|casual|technical|marketing",
            "target_audience": "general|professionals|investors|students",
            "key_sections": ["section1", "section2"],
            "writing_style": "descriptive analysis",
            "content_gaps": ["missing_topic1", "missing_topic2"],
            "strengths": ["strength1", "strength2"],
            "word_count_estimate": 1000,
            "readability_level": "intermediate|beginner|advanced"
        }}
        """
        
        response = model.generate_content(prompt)
        st.session_state.api_usage['gemini_calls'] += 1
        
        json_text = response.text.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        json_text = json_text.strip()

        analysis = json.loads(json_text)
        return analysis
        
    except Exception as e:
        st.error(f"Error analyzing content: {e}")
        return None

def generate_enhancement_suggestions(content, analysis):
    """Generate specific enhancement suggestions for the content"""
    try:
        prompt = f"""
        Based on this content analysis, provide specific enhancement suggestions:

        Content Analysis: {json.dumps(analysis, indent=2)}
        Content Sample: {content[:2000]}...

        Provide suggestions in this JSON format:
        {{
            "missing_data_points": [
                {{
                    "category": "Statistics",
                    "suggestion": "Add current market statistics",
                    "example": "Include 2024 growth percentages",
                    "priority": "high|medium|low",
                    "reasoning": "Why this is important"
                }}
            ],
            "topic_expansions": [
                {{
                    "topic": "Topic to expand",
                    "current_coverage": "brief|moderate|detailed",
                    "suggested_additions": ["addition1", "addition2"],
                    "potential_subtopics": ["subtopic1", "subtopic2"]
                }}
            ],
            "content_improvements": [
                {{
                    "area": "Structure|Data|Examples|Sources",
                    "improvement": "Specific improvement",
                    "implementation": "How to implement",
                    "impact": "Expected impact"
                }}
            ],
            "seo_enhancements": [
                {{
                    "type": "Keywords|Headers|Meta",
                    "suggestion": "Specific SEO suggestion",
                    "implementation": "How to implement"
                }}
            ],
            "fact_check_needed": [
                {{
                    "claim": "Claim that needs verification",
                    "reason": "Why it needs checking",
                    "priority": "high|medium|low"
                }}
            ]
        }}
        """
        
        response = model.generate_content(prompt)
        st.session_state.api_usage['gemini_calls'] += 1
        
        json_text = response.text.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        json_text = json_text.strip()

        suggestions = json.loads(json_text)
        return suggestions
        
    except Exception as e:
        st.error(f"Error generating suggestions: {e}")
        return None

# ========================================
# MAIN FEATURE: URL CONTENT ANALYSIS
# ========================================

st.markdown("""
<div class="url-input-section">
    <h2>🔗 NEW: URL Content Analysis & Enhancement</h2>
    <p><strong>Analyze any article URL and get AI-powered enhancement suggestions!</strong></p>
</div>
""", unsafe_allow_html=True)

# URL Input Section
st.subheader("🌐 Analyze Article from URL")

col1, col2 = st.columns([3, 1])

with col1:
    url_input = st.text_input(
        "📎 Enter Article URL to Analyze:",
        placeholder="https://example.com/article-to-analyze",
        help="Paste the full URL of any article you want to analyze and enhance"
    )

with col2:
    use_fallback = st.checkbox("Manual fallback", help="Check if URL scraping fails")

if url_input:
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Analyze URL Content", type="primary", help="Scrape and analyze the article"):
            with st.spinner("🔄 Scraping and analyzing content..."):
                # Try to scrape URL
                scraped_data, error = scrape_url_content(url_input, use_fallback)
                
                if scraped_data:
                    content = scraped_data['content']
                    metadata = scraped_data['metadata']
                    
                    # Display metadata
                    st.success("✅ Content successfully scraped and analyzed!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Word Count", metadata['word_count'])
                    with col2:
                        st.metric("Characters", metadata['char_count'])
                    with col3:
                        st.metric("Title", "✅" if metadata['title'] != 'No title found' else "❌")
                    with col4:
                        st.metric("Status", "✅ Scraped")
                    
                    # Store content for analysis
                    st.session_state.scraped_content = content
                    st.session_state.content_metadata = metadata
                    
                    # Show content preview
                    with st.expander("👀 Scraped Content Preview"):
                        st.markdown(f"**Title:** {metadata['title']}")
                        st.markdown(f"**Source:** {metadata['url']}")
                        st.text_area("Content Preview", content[:1000] + "..." if len(content) > 1000 else content, height=200, disabled=True)
                    
                    # Show immediate results first, then offer analysis
                    st.info("✅ Content scraped successfully! Click 'Analyze Content' below for AI enhancement suggestions.")
                    
                    # Immediate data extraction (fast)
                    data_points = extract_data_points(content)
                    if data_points:
                        st.subheader("📊 Quick Data Points Found")
                        for dp in data_points[:5]:  # Show first 5 quickly
                            st.markdown(f"• **{dp['value']}** ({dp['type']}): {dp['description'][:100]}...")
                    
                    # Offer detailed analysis as separate action
                    if st.button("🧠 Perform AI Analysis", type="secondary", help="Get detailed AI enhancement suggestions"):
                        # Progress tracking like original Qforia
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Step 1: Structure Analysis
                        status_text.text("🔍 Analyzing content structure...")
                        progress_bar.progress(25)
                        
                        analysis = analyze_content_structure(content)
                        
                        if analysis:
                            st.session_state.content_analysis = analysis
                            progress_bar.progress(50)
                            
                            # Step 2: Enhancement suggestions
                            status_text.text("💡 Generating enhancement suggestions...")
                            progress_bar.progress(75)
                            
                            suggestions = generate_enhancement_suggestions(content, analysis)
                            if suggestions:
                                st.session_state.enhancement_suggestions = suggestions
                                progress_bar.progress(100)
                                status_text.text("✅ Analysis complete!")
                                
                                # Clear progress after 2 seconds
                                time.sleep(1)
                                progress_bar.empty()
                                status_text.empty()
                                
                                st.success("🎉 AI analysis completed! Scroll down to see enhancement suggestions.")
                                st.rerun()
                
                else:
                    st.error(f"❌ Failed to scrape content: {error}")
                    st.info("💡 Please use the manual input option below")
    
    with col2:
        if st.button("📝 Use Manual Input", help="Paste content manually if URL scraping fails"):
            st.session_state.show_manual_input = True

# Manual Content Input (Fallback)
if 'show_manual_input' in st.session_state or (url_input and use_fallback):
    st.markdown("---")
    st.subheader("📝 Manual Content Input")
    st.info("If URL scraping doesn't work, paste the content manually:")
    
    fallback_content = st.text_area(
        "Paste article content here:",
        height=300,
        placeholder="Paste the complete article content here for analysis..."
    )
    
    if st.button("📊 Analyze Pasted Content", type="secondary") and fallback_content:
        # Immediate processing like original Qforia
        st.session_state.scraped_content = fallback_content
        st.session_state.content_metadata = {
            'word_count': len(fallback_content.split()),
            'char_count': len(fallback_content),
            'title': 'Manual Input',
            'url': url_input or 'Manual Input',
            'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Quick data extraction (immediate)
        data_points = extract_data_points(fallback_content)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Word Count", len(fallback_content.split()))
        with col2:
            st.metric("Data Points", len(data_points))
        with col3:
            st.metric("Status", "✅ Ready")
        
        st.success("✅ Content processed! Use 'Perform AI Analysis' button below for detailed suggestions.")
        
        if data_points:
            st.subheader("📊 Quick Data Points Found")
            for dp in data_points[:5]:
                st.markdown(f"• **{dp['value']}** ({dp['type']})")
        
        # Separate AI analysis button
        if st.button("🧠 Perform Detailed AI Analysis", type="secondary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔍 Analyzing content structure...")
            progress_bar.progress(33)
            
            analysis = analyze_content_structure(fallback_content)
            if analysis:
                st.session_state.content_analysis = analysis
                progress_bar.progress(66)
                
                status_text.text("💡 Generating enhancement suggestions...")
                suggestions = generate_enhancement_suggestions(fallback_content, analysis)
                if suggestions:
                    st.session_state.enhancement_suggestions = suggestions
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    st.rerun()

# Display Analysis Results
if 'content_analysis' in st.session_state and st.session_state.content_analysis:
    st.markdown("---")
    st.header("📊 AI Content Analysis Results")
    
    analysis = st.session_state.content_analysis
    
    # Show analysis completion metrics like original Qforia
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Analysis", "✅ Complete")
    with col2:
        st.metric("Content Type", analysis.get('content_type', 'N/A'))
    with col3:
        st.metric("Readability", analysis.get('readability_level', 'N/A'))
    with col4:
        st.metric("Topics Found", len(analysis.get('main_topics', [])))
    
    # Quick overview - no heavy processing
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📝 Main Topics**")
        topics = analysis.get('main_topics', [])
        for i, topic in enumerate(topics[:5], 1):
            st.markdown(f"{i}. {topic}")
    
    with col2:
        st.markdown("**🎯 Content Strengths**")
        strengths = analysis.get('strengths', [])
        for strength in strengths[:3]:
            st.markdown(f"✅ {strength}")

# Display Enhancement Suggestions (only if generated)
if 'enhancement_suggestions' in st.session_state and st.session_state.enhancement_suggestions:
    st.markdown("---")
    st.header("🚀 AI Enhancement Suggestions")
    
    suggestions = st.session_state.enhancement_suggestions
    
    # Quick metrics like original Qforia style
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        missing_count = len(suggestions.get('missing_data_points', []))
        st.metric("Missing Data", missing_count)
    with col2:
        expansion_count = len(suggestions.get('topic_expansions', []))
        st.metric("Expansions", expansion_count)
    with col3:
        improvement_count = len(suggestions.get('content_improvements', []))
        st.metric("Improvements", improvement_count)
    with col4:
        seo_count = len(suggestions.get('seo_enhancements', []))
        st.metric("SEO Fixes", seo_count)
    
    # Show suggestions in expandable format for faster loading
    # Missing Data Points
    missing_data = suggestions.get('missing_data_points', [])
    if missing_data:
        with st.expander(f"📊 Missing Data Points ({len(missing_data)})", expanded=True):
            for i, data_point in enumerate(missing_data[:3]):  # Show first 3 by default
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(data_point.get('priority', 'medium'), "🟡")
                st.markdown(f"""
                **{priority_emoji} {data_point.get('category', 'Data Point')}**  
                💡 {data_point.get('suggestion', '')}  
                📝 Example: {data_point.get('example', '')}
                """)
            
            if len(missing_data) > 3:
                if st.button(f"Show {len(missing_data) - 3} more data points"):
                    for data_point in missing_data[3:]:
                        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(data_point.get('priority', 'medium'), "🟡")
                        st.markdown(f"""
                        **{priority_emoji} {data_point.get('category', 'Data Point')}**  
                        💡 {data_point.get('suggestion', '')}  
                        📝 Example: {data_point.get('example', '')}
                        """)
    
    # Topic Expansions
    topic_expansions = suggestions.get('topic_expansions', [])
    if topic_expansions:
        with st.expander(f"📈 Topic Expansion Opportunities ({len(topic_expansions)})"):
            for expansion in topic_expansions:
                coverage_emoji = {"brief": "📝", "moderate": "📄", "detailed": "📚"}.get(expansion.get('current_coverage', 'unknown'), "📝")
                st.markdown(f"""
                **{coverage_emoji} {expansion.get('topic', '')}** ({expansion.get('current_coverage', 'unknown')} coverage)
                """)
                additions = expansion.get('suggested_additions', [])
                for addition in additions[:2]:  # Show first 2
                    st.markdown(f"• {addition}")
    
    # Quick export options (like original Qforia)
    st.subheader("📤 Quick Export")
    col1, col2 = st.columns(2)
    
    with col1:
        # Simple summary export
        summary = f"""Content Analysis Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Source: {st.session_state.content_metadata.get('url', 'Manual Input')}

Missing Data Points: {len(missing_data)}
Topic Expansions: {len(topic_expansions)}
Content Improvements: {len(suggestions.get('content_improvements', []))}
SEO Enhancements: {len(suggestions.get('seo_enhancements', []))}

Top 3 Missing Data Points:
"""
        for i, dp in enumerate(missing_data[:3], 1):
            summary += f"{i}. {dp.get('suggestion', '')} - {dp.get('example', '')}\n"
        
        st.download_button(
            "📋 Quick Summary",
            data=summary,
            file_name=f"content_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )
    
    with col2:
        # JSON export for full data
        enhancement_report = {
            'content_metadata': st.session_state.content_metadata,
            'analysis': st.session_state.content_analysis,
            'suggestions': st.session_state.enhancement_suggestions,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        report_json = json.dumps(enhancement_report, indent=2, default=str)
        st.download_button(
            "📋 Full Report JSON",
            data=report_json,
            file_name=f"content_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )

# ========================================
# OTHER FEATURES
# ========================================

st.markdown("---")
st.header("🔧 Additional Features")

# Feature Selection
feature_selection = st.selectbox(
    "Choose additional feature:",
    ["Select a feature...", "🔍 Query Fan-Out", "🔬 Research & Fact-Check", "📊 Quick Fact Checker"]
)

if feature_selection == "🔍 Query Fan-Out":
    st.subheader("🔍 Query Fan-Out Simulator")
    
    # Query Fan-Out Functions
    def QUERY_FANOUT_PROMPT(q, mode):
        min_queries_simple = 10
        min_queries_complex = 20

        if mode == "AI Overview (simple)":
            target = min_queries_simple
            instruction = f"Generate {min_queries_simple}-{min_queries_simple + 2} queries for a simple overview"
        else:
            target = min_queries_complex
            instruction = f"Generate {min_queries_complex}-{min_queries_complex + 5} queries for comprehensive analysis"

        return f"""
You are simulating Google's AI Mode query fan-out process.
Original query: "{q}"
Mode: "{mode}"

{instruction}

Include these query types:
1. Reformulations
2. Related Queries  
3. Implicit Queries
4. Comparative Queries
5. Entity Expansions
6. Personalized Queries

Return only valid JSON:
{{
  "generation_details": {{
    "target_query_count": {target},
    "reasoning_for_count": "Brief reasoning for number of queries"
  }},
  "expanded_queries": [
    {{
      "query": "Example query",
      "type": "reformulation",
      "user_intent": "Intent description",
      "reasoning": "Why this query was generated"
    }}
  ]
}}
"""

    def generate_fanout(query, mode):
        """Generate query fan-out using Gemini"""
        prompt = QUERY_FANOUT_PROMPT(query, mode)
        try:
            response = model.generate_content(prompt)
            st.session_state.api_usage['gemini_calls'] += 1
            
            json_text = response.text.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            json_text = json_text.strip()

            data = json.loads(json_text)
            return data.get("expanded_queries", []), data.get("generation_details", {})
        except Exception as e:
            st.error(f"Error generating fan-out: {e}")
            return None, None

    col1, col2 = st.columns([3, 1])

    with col1:
        user_query = st.text_area("Enter your query", "Why to Invest in Bangalore", height=100)
        mode = st.radio("Search Mode", ["AI Overview (simple)", "AI Mode (complex)"])

    with col2:
        st.markdown("**Current Usage:**")
        st.metric("Gemini", st.session_state.api_usage['gemini_calls'])
        st.metric("Perplexity", st.session_state.api_usage['perplexity_calls'])

    if st.button("🚀 Generate Query Fan-Out", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Generating query fan-out..."):
                results, details = generate_fanout(user_query, mode)

            if results:
                st.session_state.fanout_results = results
                st.success(f"✅ Generated {len(results)} queries!")

                if details:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Target Queries", details.get('target_query_count', 'N/A'))
                    with col2:
                        st.metric("Generated", len(results))
                    with col3:
                        st.metric("Success Rate", "100%")
                    
                    st.info(f"**Reasoning:** {details.get('reasoning_for_count', 'Not provided')}")

                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download CSV", data=csv, file_name="fanout_queries.csv", mime="text/csv")
                with col2:
                    json_data = df.to_json(orient='records', indent=2)
                    st.download_button("📥 Download JSON", data=json_data, file_name="fanout_queries.json", mime="application/json")

elif feature_selection == "🔬 Research & Fact-Check":
    st.subheader("🔬 Research & Fact-Check Queries")
    
    # Research source selection
    research_source = st.radio(
        "Research based on:",
        ["🔍 Fan-out queries", "📄 Content analysis suggestions", "✍️ Custom queries"]
    )
    
    selected_queries = []
    
    if research_source == "🔍 Fan-out queries":
        if st.session_state.fanout_results:
            query_options = [f"{i+1}. {q['query']}" for i, q in enumerate(st.session_state.fanout_results)]
            selected_queries = st.multiselect(
                "Select queries to research (max 10):",
                options=query_options,
                default=query_options[:5] if len(query_options) > 5 else query_options
            )
        else:
            st.info("No fan-out queries available. Please generate queries first.")
    
    elif research_source == "📄 Content analysis suggestions":
        if 'enhancement_suggestions' in st.session_state and st.session_state.enhancement_suggestions:
            suggestions = st.session_state.enhancement_suggestions
            
            # Create research queries from enhancement suggestions
            research_queries = []
            
            # From missing data points
            for data_point in suggestions.get('missing_data_points', []):
                query = f"Latest data on {data_point.get('category', '')}: {data_point.get('suggestion', '')}"
                research_queries.append(query)
            
            # From topic expansions
            for expansion in suggestions.get('topic_expansions', []):
                for addition in expansion.get('suggested_additions', []):
                    query = f"Research {expansion.get('topic', '')}: {addition}"
                    research_queries.append(query)
            
            if research_queries:
                selected_queries = st.multiselect(
                    "Select enhancement-based research queries (max 10):",
                    options=research_queries,
                    default=research_queries[:5] if len(research_queries) > 5 else research_queries
                )
            else:
                st.info("No content analysis suggestions available. Please analyze content first.")
        else:
            st.info("No content analysis available. Please analyze content first.")
    
    else:  # Custom queries
        custom_queries_text = st.text_area(
            "Enter custom research queries (one per line):",
            placeholder="Latest Bangalore real estate prices 2024\nBangalore infrastructure development projects\nBangalore job market growth statistics",
            height=150
        )
        
        if custom_queries_text:
            custom_queries = [q.strip() for q in custom_queries_text.split('\n') if q.strip()]
            selected_queries = st.multiselect(
                "Select custom queries to research:",
                options=custom_queries,
                default=custom_queries[:5] if len(custom_queries) > 5 else custom_queries
            )
    
    # Research execution
    if selected_queries:
        research_focus = st.selectbox("Research Focus", [
            "Market Data & Statistics",
            "Current Facts & Trends", 
            "Investment Information",
            "Comparative Analysis",
            "Growth & Financial Data",
            "Content Enhancement Data"
        ])
        
        if st.button("🔍 Start Research & Fact-Check", type="secondary"):
            if len(selected_queries) > 10:
                st.warning("Limited to 10 queries to control API costs.")
            else:
                research_results = []
                progress_bar = st.progress(0)
                
                for i, query in enumerate(selected_queries):
                    # Extract clean query text
                    if query.startswith(tuple("123456789")):
                        query_text = query.split('. ', 1)[1] if '. ' in query else query
                    else:
                        query_text = query
                    
                    progress_bar.progress((i + 1) / len(selected_queries))
                    
                    with st.spinner(f"Researching: {query_text[:50]}..."):
                        research_prompt = f"""
                        Research this query focusing on {research_focus}: {query_text}
                        
                        Provide:
                        1. Key facts with specific numbers and statistics
                        2. Current market data and trends
                        3. Recent developments and changes
                        4. Credible sources and references
                        
                        Focus on actionable information with numerical data.
                        """
                        
                        response = call_perplexity_api(research_prompt)
                        
                        if 'choices' in response:
                            content = response['choices'][0]['message']['content']
                            data_points = extract_data_points(content)
                            
                            research_results.append({
                                'query': query_text,
                                'research_content': content,
                                'data_points': data_points,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
                            })
                        else:
                            research_results.append({
                                'query': query_text,
                                'research_content': f"Error: {response.get('error', 'Unknown error')}",
                                'data_points': [],
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
                            })
                    
                    time.sleep(1)  # Rate limiting
                
                progress_bar.progress(1.0)
                st.session_state.research_data = research_results
                
                # Display results
                st.success(f"✅ Research completed for {len(research_results)} queries!")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Queries Researched", len(research_results))
                with col2:
                    total_data_points = sum(len(r['data_points']) for r in research_results)
                    st.metric("Data Points Found", total_data_points)
                with col3:
                    successful = sum(1 for r in research_results if 'Error:' not in r['research_content'])
                    st.metric("Success Rate", f"{(successful/len(research_results)*100):.0f}%")
                
                # Research Results Display
                for i, result in enumerate(research_results):
                    with st.expander(f"📋 {result['query'][:80]}..."):
                        st.markdown("**Research Findings:**")
                        st.markdown(result['research_content'])
                        
                        if result['data_points']:
                            st.markdown("**📊 Key Data Points:**")
                            
                            df_data = []
                            for dp in result['data_points']:
                                df_data.append({
                                    'Value': dp['value'],
                                    'Type': dp['type'],
                                    'Description': dp['description']
                                })
                            
                            if df_data:
                                data_df = pd.DataFrame(df_data)
                                st.dataframe(data_df, hide_index=True, use_container_width=True)
                        
                        st.caption(f"Researched on: {result['timestamp']}")

elif feature_selection == "📊 Quick Fact Checker":
    st.subheader("🔍 Quick Fact Checker")

    def call_perplexity_answer_api(query):
        """Call Perplexity Answer API for fact-checking"""
        try:
            headers = {
                "Authorization": f"Bearer {perplexity_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "sonar-pro",
                "messages": [
                    {"role": "user", "content": f"Please provide a factual answer with sources for: {query}"}
                ],
                "temperature": 0.1,
                "max_tokens": 800
            }
            
            response = requests.post("https://api.perplexity.ai/chat/completions", 
                                   headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                st.session_state.api_usage['perplexity_calls'] += 1
                result = response.json()
                
                if 'choices' in result and result['choices']:
                    answer = result['choices'][0]['message']['content']
                    return {"answer": answer, "sources": []}
                else:
                    return {"error": "No answer received"}
            else:
                error_details = response.text if response.text else "No error details"
                return {"error": f"API call failed with status {response.status_code}. Details: {error_details}"}
        
        except Exception as e:
            return {"error": f"Exception occurred: {str(e)}"}

    fact_query = st.text_input(
        "Enter a statement or topic to fact-check:", 
        placeholder="e.g., Bangalore property prices increased by 15% in 2024"
    )

    if st.button("🔍 Verify Facts") and fact_query:
        with st.spinner("Fact-checking via Perplexity..."):
            response = call_perplexity_answer_api(fact_query)

            if 'answer' in response:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📋 Fact-Based Answer")
                    st.markdown(response['answer'])
                    
                    data_points = extract_data_points(response['answer'])
                    
                    if data_points:
                        st.subheader("📊 Extracted Data Points")
                        df_data = []
                        for dp in data_points:
                            df_data.append({
                                'Value': dp['value'],
                                'Type': dp['type'],
                                'Description': dp['description']
                            })
                        
                        fact_df = pd.DataFrame(df_data)
                        st.dataframe(fact_df, hide_index=True, use_container_width=True)
                
                with col2:
                    st.subheader("📊 Summary")
                    st.metric("Data Points Found", len(data_points))
                    if data_points:
                        for dp in data_points[:3]:
                            st.markdown(f"• **{dp['value']}** ({dp['type']})")
            else:
                st.error(response.get("error", "Unknown error."))

# Footer
st.markdown("---")
st.subheader("🎯 Quick Actions & Help")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🗑️ Clear All Data"):
        # Clear all session state
        for key in ['fanout_results', 'research_data', 'content_analysis', 'enhancement_suggestions', 'scraped_content', 'content_metadata', 'show_manual_input']:
            if key in st.session_state:
                del st.session_state[key]
        st.success("All data cleared!")
        st.rerun()

with col2:
    if st.button("💡 Usage Tips"):
        st.info("""
        **Pro Tips:**
        1. Start with URL analysis for quick content insights
        2. Use manual input if URL scraping fails
        3. Generate research queries from content analysis
        4. Export data for offline analysis
        5. Cross-reference facts with fact-checker
        """)

with col3:
    if st.button("🔧 Troubleshooting"):
        st.info("""
        **Common Issues:**
        1. **URL Scraping Fails:** Use manual content input
        2. **API Errors:** Check API keys and rate limits
        3. **No Analysis Results:** Content might be too short
        4. **Rate Limited:** Wait and retry
        """)

with col4:
    if st.button("📊 View Session Stats"):
        features_used = []
        if st.session_state.fanout_results:
            features_used.append("Query Fan-out")
        if 'content_analysis' in st.session_state:
            features_used.append("Content Analysis")
        if st.session_state.research_data:
            features_used.append("Research")
        
        st.info(f"""
        **Session Statistics:**
        - Features Used: {len(features_used)}
        - Active Features: {', '.join(features_used) if features_used else 'None'}
        - Total API Calls: {st.session_state.api_usage['gemini_calls'] + st.session_state.api_usage['perplexity_calls']}
        - Estimated Cost: ${estimated_cost:.3f}
        """)

st.markdown("---")
st.markdown("**Qforia Pro Enhanced v3.0** - Advanced Query Research & Content Analysis Tool")
st.markdown("*🌟 NEW: URL Content Analysis with AI-Powered Enhancement Suggestions*")
