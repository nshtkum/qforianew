import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import requests
import time
from datetime import datetime
from bs4 import BeautifulSoup

# App config
st.set_page_config(page_title="Qforia Research Platform", layout="wide")
st.title("🔍 Qforia: Complete Research & Analysis Platform")

# Initialize session states
if 'fanout_results' not in st.session_state:
    st.session_state.fanout_results = None
if 'generation_details' not in st.session_state:
    st.session_state.generation_details = None
if 'research_results' not in st.session_state:
    st.session_state.research_results = {}
if 'selected_queries' not in st.session_state:
    st.session_state.selected_queries = set()
if 'url_analysis' not in st.session_state:
    st.session_state.url_analysis = None
if 'enhanced_topics' not in st.session_state:
    st.session_state.enhanced_topics = []

# API Configuration using Streamlit secrets
try:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    perplexity_key = st.secrets["PERPLEXITY_API_KEY"]
    
    # Configure Gemini
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    
    # Show configuration status in sidebar
    st.sidebar.header("🔧 Configuration Status")
    st.sidebar.success("✅ Gemini API Key: Configured")
    st.sidebar.success("✅ Perplexity API Key: Configured")
    
except KeyError as e:
    st.sidebar.header("🔧 Configuration")
    st.sidebar.error(f"❌ Missing API key in secrets: {e}")
    st.sidebar.info("Please add the following to your Streamlit secrets:")
    st.sidebar.code('''
[secrets]
GEMINI_API_KEY = "your_gemini_key_here"
PERPLEXITY_API_KEY = "your_perplexity_key_here"
    ''')
    
    # Fallback to manual input
    st.sidebar.subheader("Manual Input (Fallback)")
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
    perplexity_key = st.sidebar.text_input("Perplexity API Key", type="password")
    
    if gemini_key:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
    else:
        model = None

# Utility Functions
def call_perplexity(query, system_prompt="Provide comprehensive, actionable insights with specific data points, statistics, and practical recommendations."):
    if not perplexity_key:
        return {"error": "Missing Perplexity API key"}
    
    headers = {
        "Authorization": f"Bearer {perplexity_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    data = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"Perplexity API error: {e}"}

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
        return {'content': text[:8000], 'title': title, 'url': url}, None
    except Exception as e:
        return None, f"Scraping error: {e}"

def analyze_url_content(content, title):
    if not model:
        return None, "Gemini not configured"
    
    try:
        prompt = f"""
        Analyze this URL content and extract key information:
        Title: {title}
        Content: {content}

        Provide a comprehensive analysis in JSON format:
        {{
            "main_topics": ["topic1", "topic2", "topic3"],
            "key_points": ["point1", "point2", "point3"],
            "content_type": "news|blog|research|commercial|educational",
            "credibility_score": "high|medium|low",
            "missing_context": [
                {{"topic": "topic", "missing_info": "what's missing", "research_query": "specific query"}}
            ],
            "fact_check_items": [
                {{"claim": "specific claim", "verification_query": "query to verify this"}}
            ],
            "enhancement_opportunities": [
                {{"area": "area to enhance", "suggested_research": "what to research"}}
            ]
        }}
        """
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        
        # Clean JSON
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        json_text = json_text.strip()
        
        return json.loads(json_text), None
    except Exception as e:
        return None, f"Analysis error: {e}"

def QUERY_FANOUT_PROMPT(q, mode):
    min_queries_simple = 12
    min_queries_complex = 25

    if mode == "AI Overview (simple)":
        num_queries_instruction = (
            f"Analyze the user's query: \"{q}\". For '{mode}' mode, "
            f"generate **at least {min_queries_simple} diverse queries** that cover: "
            f"basic information, comparisons, alternatives, practical considerations, and user scenarios. "
            f"Focus on queries that would provide comprehensive coverage for someone researching this topic."
        )
    else:  # AI Mode (complex)
        num_queries_instruction = (
            f"Analyze the user's query: \"{q}\". For '{mode}' mode, "
            f"generate **at least {min_queries_complex} comprehensive queries** that include: "
            f"deep analysis, market trends, technical specifications, expert opinions, case studies, "
            f"future predictions, regulatory considerations, and advanced comparisons. "
            f"Create queries suitable for exhaustive research and strategic decision-making."
        )

    return (
        f"You are an expert research strategist creating a comprehensive query fan-out for: \"{q}\"\n"
        f"Mode: {mode}\n\n"
        f"{num_queries_instruction}\n\n"
        f"Create queries across these categories (ensure good distribution):\n"
        f"1. **Core Information**: Direct answers and fundamental concepts\n"
        f"2. **Comparative Analysis**: Comparisons with alternatives and competitors\n"
        f"3. **Market Intelligence**: Trends, statistics, market dynamics\n"
        f"4. **Technical Deep-Dive**: Specifications, features, capabilities\n"
        f"5. **User Experience**: Reviews, testimonials, real-world usage\n"
        f"6. **Strategic Considerations**: Cost analysis, ROI, decision factors\n"
        f"7. **Future Outlook**: Predictions, upcoming developments\n"
        f"8. **Expert Insights**: Professional opinions, industry analysis\n\n"
        f"Return ONLY valid JSON in this exact format:\n"
        f"{{\n"
        f"  \"generation_details\": {{\n"
        f"    \"target_query_count\": <number>,\n"
        f"    \"reasoning_for_count\": \"<explanation>\",\n"
        f"    \"research_strategy\": \"<overall approach>\"\n"
        f"  }},\n"
        f"  \"expanded_queries\": [\n"
        f"    {{\n"
        f"      \"query\": \"<specific research question>\",\n"
        f"      \"category\": \"<one of the 8 categories above>\",\n"
        f"      \"priority\": \"<high/medium/low>\",\n"
        f"      \"expected_insights\": \"<what this query should reveal>\",\n"
        f"      \"research_value\": \"<why this is important for decision-making>\"\n"
        f"    }}\n"
        f"  ]\n"
        f"}}"
    )

def generate_fanout(query, mode):
    if not model:
        st.error("Please configure Gemini API key")
        return None
        
    prompt = QUERY_FANOUT_PROMPT(query, mode)
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        
        # Clean JSON
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        json_text = json_text.strip()

        data = json.loads(json_text)
        return data
    except Exception as e:
        st.error(f"Error generating fanout: {e}")
        return None

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Query Research", "🌐 URL Analyzer", "✅ Fact Checker", "📊 Research Dashboard"])

with tab1:
    st.header("🎯 Qforia Query Fan-Out Research")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Your Research Query")
        user_query = st.text_area(
            "What would you like to research?", 
            value="What's the best electric SUV for driving up Mt. Rainier?",
            height=100,
            help="Enter any topic you want to research comprehensively"
        )

    with col2:
        st.subheader("Research Settings")
        mode = st.selectbox(
            "Research Depth",
            ["AI Overview (simple)", "AI Mode (complex)"],
            help="Simple: 12+ focused queries | Complex: 25+ comprehensive queries"
        )
        
        if st.button("🚀 Generate Research Queries", type="primary", use_container_width=True):
            if not user_query.strip():
                st.warning("Please enter a research query")
            elif not model:
                st.warning("Please configure your API keys in Streamlit secrets")
            else:
                with st.spinner("🤖 Generating comprehensive research queries..."):
                    results = generate_fanout(user_query, mode)
                    
                if results:
                    st.session_state.fanout_results = results
                    st.session_state.generation_details = results.get("generation_details", {})
                    st.session_state.selected_queries = set()
                    st.success("✅ Research queries generated successfully!")
                    st.rerun()

    # Display fanout results
    if st.session_state.fanout_results:
        st.markdown("---")
        
        # Show generation details
        details = st.session_state.generation_details
        if details:
            st.subheader("🧠 Research Strategy")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Target Queries", details.get('target_query_count', 'N/A'))
            with col2:
                st.metric("Generated", len(st.session_state.fanout_results.get('expanded_queries', [])))
            with col3:
                if perplexity_key:
                    st.metric("Research Ready", "✅")
                else:
                    st.metric("Research Ready", "❌ Need Perplexity Key")
            
            st.info(f"**Strategy:** {details.get('research_strategy', 'Not provided')}")

        # Interactive query selection
        st.subheader("📋 Research Queries - Select for Deep Research")
        
        queries = st.session_state.fanout_results.get('expanded_queries', [])
        if queries:
            # Category filter
            categories = list(set(q.get('category', 'Unknown') for q in queries))
            selected_categories = st.multiselect(
                "Filter by Category:", 
                categories, 
                default=categories
            )
            
            # Priority filter
            priorities = list(set(q.get('priority', 'medium') for q in queries))
            selected_priorities = st.multiselect(
                "Filter by Priority:", 
                priorities, 
                default=priorities
            )
            
            # Filter queries
            filtered_queries = [
                q for q in queries 
                if q.get('category', 'Unknown') in selected_categories 
                and q.get('priority', 'medium') in selected_priorities
            ]
            
            st.write(f"Showing {len(filtered_queries)} of {len(queries)} queries")
            
            # Display queries with selection
            for i, query_data in enumerate(filtered_queries):
                query_id = f"query_{hash(query_data['query'])}"
                
                with st.container():
                    col1, col2, col3 = st.columns([1, 6, 2])
                    
                    with col1:
                        selected = st.checkbox("Select", key=f"checkbox_{query_id}")
                        if selected:
                            st.session_state.selected_queries.add(query_id)
                        else:
                            st.session_state.selected_queries.discard(query_id)
                    
                    with col2:
                        priority_color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(query_data.get('priority', 'medium'), '🟡')
                        st.markdown(f"**{priority_color} {query_data['query']}**")
                        st.caption(f"📁 {query_data.get('category', 'Unknown')} | 💡 {query_data.get('expected_insights', 'N/A')}")
                    
                    with col3:
                        if query_id in st.session_state.research_results:
                            st.success("✅ Researched")
                        elif perplexity_key:
                            if st.button("🔍 Research", key=f"research_{query_id}"):
                                with st.spinner("Researching..."):
                                    result = call_perplexity(query_data['query'])
                                    if 'choices' in result:
                                        st.session_state.research_results[query_id] = {
                                            'query': query_data['query'],
                                            'result': result['choices'][0]['message']['content'],
                                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                                            'category': query_data.get('category', 'Unknown'),
                                            'priority': query_data.get('priority', 'medium')
                                        }
                                        st.rerun()
                        else:
                            st.caption("Need Perplexity Key")
                            
                    st.markdown("---")
            
            # Bulk research
            if st.session_state.selected_queries and perplexity_key:
                if st.button("🚀 Research Selected Queries", type="secondary"):
                    selected_query_data = [
                        q for q in filtered_queries 
                        if f"query_{hash(q['query'])}" in st.session_state.selected_queries
                    ]
                    
                    progress_bar = st.progress(0)
                    for i, query_data in enumerate(selected_query_data):
                        query_id = f"query_{hash(query_data['query'])}"
                        if query_id not in st.session_state.research_results:
                            result = call_perplexity(query_data['query'])
                            if 'choices' in result:
                                st.session_state.research_results[query_id] = {
                                    'query': query_data['query'],
                                    'result': result['choices'][0]['message']['content'],
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                                    'category': query_data.get('category', 'Unknown'),
                                    'priority': query_data.get('priority', 'medium')
                                }
                            time.sleep(1)
                        progress_bar.progress((i + 1) / len(selected_query_data))
                    st.success("✅ Bulk research completed!")
                    st.rerun()

with tab2:
    st.header("🌐 URL Content Analyzer")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        url = st.text_input("Enter URL to analyze:", placeholder="https://example.com/article")
        
    with col2:
        if st.button("🔍 Analyze URL", type="primary", use_container_width=True):
            if not url:
                st.warning("Please enter a URL")
            elif not model:
                st.warning("Please configure your API keys in Streamlit secrets")
            else:
                with st.spinner("🌐 Scraping and analyzing URL content..."):
                    scraped_data, error = scrape_url(url)
                    
                if scraped_data:
                    st.success("✅ Content scraped successfully!")
                    st.write(f"**Title:** {scraped_data['title']}")
                    
                    with st.spinner("🤖 Analyzing content..."):
                        analysis, error = analyze_url_content(scraped_data['content'], scraped_data['title'])
                    
                    if analysis:
                        st.session_state.url_analysis = analysis
                        st.success("✅ Analysis completed!")
                        st.rerun()
                    else:
                        st.error(f"Analysis failed: {error}")
                else:
                    st.error(f"Scraping failed: {error}")

    # Display URL analysis results
    if st.session_state.url_analysis:
        st.markdown("---")
        analysis = st.session_state.url_analysis
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Content Type", analysis.get('content_type', 'Unknown').title())
        with col2:
            st.metric("Credibility", analysis.get('credibility_score', 'Unknown').title())
        with col3:
            st.metric("Topics Found", len(analysis.get('main_topics', [])))
        with col4:
            st.metric("Enhancement Ops", len(analysis.get('enhancement_opportunities', [])))
        
        # Main topics
        if analysis.get('main_topics'):
            st.subheader("📝 Main Topics")
            for topic in analysis['main_topics']:
                st.write(f"• {topic}")
        
        # Key points
        if analysis.get('key_points'):
            st.subheader("🎯 Key Points")
            for point in analysis['key_points']:
                st.write(f"• {point}")
        
        # Missing context & enhancement opportunities
        col1, col2 = st.columns(2)
        
        with col1:
            if analysis.get('missing_context'):
                st.subheader("❓ Missing Context")
                for item in analysis['missing_context']:
                    with st.expander(f"📌 {item['topic']}"):
                        st.write(f"**Missing:** {item['missing_info']}")
                        if perplexity_key and st.button(f"Research: {item['topic']}", key=f"missing_{hash(item['topic'])}"):
                            result = call_perplexity(item['research_query'])
                            if 'choices' in result:
                                st.write("**Research Result:**")
                                st.write(result['choices'][0]['message']['content'])
        
        with col2:
            if analysis.get('enhancement_opportunities'):
                st.subheader("🚀 Enhancement Opportunities")
                for item in analysis['enhancement_opportunities']:
                    with st.expander(f"💡 {item['area']}"):
                        st.write(f"**Suggested Research:** {item['suggested_research']}")
                        if perplexity_key and st.button(f"Enhance: {item['area']}", key=f"enhance_{hash(item['area'])}"):
                            result = call_perplexity(item['suggested_research'])
                            if 'choices' in result:
                                st.write("**Enhancement Data:**")
                                st.write(result['choices'][0]['message']['content'])

with tab3:
    st.header("✅ Fact Checker & Claim Verification")
    
    # Manual fact checking
    st.subheader("🔍 Manual Fact Check")
    fact_query = st.text_input("Enter claim to verify:", placeholder="e.g., Tesla Model Y is the best-selling EV in 2024")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔍 Verify Fact", type="primary"):
            if fact_query and perplexity_key:
                with st.spinner("Verifying claim..."):
                    verification_prompt = f"Fact-check this claim with current data and sources: {fact_query}. Provide verification status, supporting evidence, and source citations."
                    result = call_perplexity(fact_query, verification_prompt)
                    if 'choices' in result:
                        st.write("**Verification Result:**")
                        st.write(result['choices'][0]['message']['content'])
            elif not perplexity_key:
                st.warning("Please enter Perplexity API key")
            else:
                st.warning("Please enter a claim to verify")
    
    # Auto fact-checking from URL analysis
    if st.session_state.url_analysis and st.session_state.url_analysis.get('fact_check_items'):
        st.markdown("---")
        st.subheader("🤖 Auto-Detected Claims for Verification")
        
        for item in st.session_state.url_analysis['fact_check_items']:
            with st.expander(f"📋 {item['claim']}", expanded=False):
                if perplexity_key:
                    if st.button(f"Verify Claim", key=f"verify_{hash(item['claim'])}"):
                        with st.spinner("Verifying..."):
                            result = call_perplexity(item['verification_query'], "Fact-check this claim with current data, sources, and verification status.")
                            if 'choices' in result:
                                st.write("**Verification Result:**")
                                st.write(result['choices'][0]['message']['content'])
                else:
                    st.caption("Perplexity API key required for verification")

with tab4:
    st.header("📊 Research Dashboard")
    
    if st.session_state.research_results:
        # Summary metrics
        total_researched = len(st.session_state.research_results)
        categories_researched = len(set(r['category'] for r in st.session_state.research_results.values()))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Researched", total_researched)
        with col2:
            st.metric("Categories Covered", categories_researched)
        with col3:
            st.metric("Research Depth", "Comprehensive" if total_researched > 10 else "Basic")
        
        # Export research results
        research_df = pd.DataFrame([
            {
                'Query': data['query'],
                'Category': data['category'],
                'Priority': data['priority'],
                'Research Findings': data['result'],
                'Timestamp': data['timestamp']
            }
            for data in st.session_state.research_results.values()
        ])
        
        csv_data = research_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Complete Research Results",
            data=csv_data,
            file_name=f"qforia_complete_research_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
        
        # Detailed results
        st.subheader("📋 Detailed Research Results")
        for query_id, data in st.session_state.research_results.items():
            with st.expander(f"🔍 {data['query']}", expanded=False):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("**Research Findings:**")
                    st.write(data['result'])
                with col2:
                    st.caption(f"**Category:** {data['category']}")
                    st.caption(f"**Priority:** {data['priority']}")
                    st.caption(f"**Researched:** {data['timestamp']}")
    else:
        st.info("No research results yet. Start by using the Query Research or URL Analyzer tabs.")

# Clear all data button
if st.sidebar.button("🗑️ Clear All Data"):
    st.session_state.fanout_results = None
    st.session_state.generation_details = None
    st.session_state.research_results = {}
    st.session_state.selected_queries = set()
    st.session_state.url_analysis = None
    st.session_state.enhanced_topics = []
    st.success("All data cleared!")
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**Qforia Complete Research Platform** - Query Fan-Out, URL Analysis, Fact Checking & Research Dashboard | *Powered by Gemini AI & Perplexity*")
