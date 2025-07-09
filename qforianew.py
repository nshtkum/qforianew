import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import requests
import time
from datetime import datetime

# App config
st.set_page_config(page_title="Qforia Enhanced", layout="wide")
st.title("🔍 Qforia: Query Fan-Out Simulator & Research Engine")

# Initialize session states
if 'fanout_results' not in st.session_state:
    st.session_state.fanout_results = None
if 'generation_details' not in st.session_state:
    st.session_state.generation_details = None
if 'research_results' not in st.session_state:
    st.session_state.research_results = {}
if 'selected_queries' not in st.session_state:
    st.session_state.selected_queries = set()

# Sidebar Configuration
st.sidebar.header("🔧 Configuration")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
perplexity_key = st.sidebar.text_input("Perplexity API Key", type="password")

# Configure Gemini
if gemini_key:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
else:
    model = None

# Perplexity API function
def call_perplexity(query):
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
            {"role": "system", "content": "Provide comprehensive, actionable insights with specific data points, statistics, and practical recommendations. Include current trends and recent developments."},
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

# Enhanced prompt for query fanout
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
        f"Each query should be:\n"
        f"- Specific and actionable for research\n"
        f"- Likely to yield unique, valuable insights\n"
        f"- Suitable for follow-up research via search engines\n\n"
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

# Generate fanout
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

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎯 Enter Your Research Query")
    user_query = st.text_area(
        "What would you like to research?", 
        value="What's the best electric SUV for driving up Mt. Rainier?",
        height=100,
        help="Enter any topic you want to research comprehensively"
    )

with col2:
    st.subheader("⚙️ Research Settings")
    mode = st.selectbox(
        "Research Depth",
        ["AI Overview (simple)", "AI Mode (complex)"],
        help="Simple: 12+ focused queries | Complex: 25+ comprehensive queries"
    )
    
    if st.button("🚀 Generate Research Queries", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("Please enter a research query")
        elif not gemini_key:
            st.warning("Please enter your Gemini API key in the sidebar")
        else:
            with st.spinner("🤖 Generating comprehensive research queries..."):
                results = generate_fanout(user_query, mode)
                
            if results:
                st.session_state.fanout_results = results
                st.session_state.generation_details = results.get("generation_details", {})
                st.session_state.selected_queries = set()  # Reset selections
                st.success("✅ Research queries generated successfully!")
                st.rerun()

# Display results
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
        
        with st.expander("View Generation Reasoning"):
            st.write(details.get('reasoning_for_count', 'Not provided'))

    # Interactive query selection
    st.subheader("📋 Research Queries - Select for Deep Research")
    
    queries = st.session_state.fanout_results.get('expanded_queries', [])
    if queries:
        # Category filter
        categories = list(set(q.get('category', 'Unknown') for q in queries))
        selected_categories = st.multiselect(
            "Filter by Category:", 
            categories, 
            default=categories,
            help="Filter queries by research category"
        )
        
        # Priority filter
        priorities = list(set(q.get('priority', 'medium') for q in queries))
        selected_priorities = st.multiselect(
            "Filter by Priority:", 
            priorities, 
            default=priorities,
            help="Filter by research priority level"
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
                    selected = st.checkbox(
                        "Select", 
                        key=f"checkbox_{query_id}",
                        value=query_id in st.session_state.selected_queries
                    )
                    if selected:
                        st.session_state.selected_queries.add(query_id)
                    else:
                        st.session_state.selected_queries.discard(query_id)
                
                with col2:
                    priority_color = {
                        'high': '🔴',
                        'medium': '🟡', 
                        'low': '🟢'
                    }.get(query_data.get('priority', 'medium'), '🟡')
                    
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
                                    st.error(f"Research failed: {result.get('error', 'Unknown error')}")
                    else:
                        st.caption("Need Perplexity Key")
                        
                st.markdown("---")
        
        # Bulk actions
        if st.session_state.selected_queries and perplexity_key:
            st.subheader("🔥 Bulk Research Actions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚀 Research Selected Queries", type="secondary"):
                    selected_query_data = [
                        q for q in filtered_queries 
                        if f"query_{hash(q['query'])}" in st.session_state.selected_queries
                    ]
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, query_data in enumerate(selected_query_data):
                        query_id = f"query_{hash(query_data['query'])}"
                        if query_id not in st.session_state.research_results:
                            status_text.text(f"Researching: {query_data['query'][:50]}...")
                            result = call_perplexity(query_data['query'])
                            
                            if 'choices' in result:
                                st.session_state.research_results[query_id] = {
                                    'query': query_data['query'],
                                    'result': result['choices'][0]['message']['content'],
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                                    'category': query_data.get('category', 'Unknown'),
                                    'priority': query_data.get('priority', 'medium')
                                }
                            time.sleep(1)  # Rate limiting
                        
                        progress_bar.progress((i + 1) / len(selected_query_data))
                    
                    status_text.text("✅ Bulk research completed!")
                    st.rerun()
            
            with col2:
                st.caption(f"{len(st.session_state.selected_queries)} queries selected")
            
            with col3:
                if st.button("🗑️ Clear Selection"):
                    st.session_state.selected_queries = set()
                    st.rerun()

# Display research results
if st.session_state.research_results:
    st.markdown("---")
    st.subheader("📊 Research Results")
    
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
    
    # Export options
    research_df = pd.DataFrame([
        {
            'Query': data['query'],
            'Category': data['category'],
            'Priority': data['priority'],
            'Research Findings': data['result'][:200] + "..." if len(data['result']) > 200 else data['result'],
            'Timestamp': data['timestamp']
        }
        for data in st.session_state.research_results.values()
    ])
    
    csv_data = research_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Research Results",
        data=csv_data,
        file_name=f"qforia_research_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
    
    # Detailed results
    for query_id, data in st.session_state.research_results.items():
        with st.expander(f"🔍 {data['query']}", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Research Findings:**")
                st.write(data['result'])
            with col2:
                st.caption(f"**Category:** {data['category']}")
                st.caption(f"**Priority:** {data['priority']}")
                st.caption(f"**Researched:** {data['timestamp']}")

# Download fanout results
if st.session_state.fanout_results:
    queries_df = pd.DataFrame(st.session_state.fanout_results.get('expanded_queries', []))
    if not queries_df.empty:
        csv_fanout = queries_df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            "📥 Download Query Fanout",
            data=csv_fanout,
            file_name=f"qforia_fanout_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("**Qforia Enhanced** - Advanced Query Fan-Out & Research Engine | *Powered by Gemini AI & Perplexity*")
