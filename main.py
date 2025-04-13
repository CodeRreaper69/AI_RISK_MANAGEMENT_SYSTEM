import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import datetime
import random
from pathlib import Path
import json
import os
import google.generativeai as genai
from textblob import TextBlob
from dotenv import load_dotenv

from utils.utils import market_impact_on_project, create_master_news_list
from utils.risk_tools import calculate_risk_score, get_risk_level, project_risk_summary, create_project_metrics_table, create_risk_factors_chart, format_published_date, create_risk_gauge

load_dotenv()


# Set page configuration
st.set_page_config(
    page_title="AI Project Risk Management System",
    page_icon="🛠️",
    layout="wide"
)

# Constants
HIGH_RISK_THRESHOLD = 70
MEDIUM_RISK_THRESHOLD = 40

# Initialize session state variables if they don't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_project' not in st.session_state:
    st.session_state.selected_project = None


# Generate sample data
def load_sample_project_data():
    """Load or generate sample project data"""
    try:
        # Try to load existing data
        df = pd.read_csv("project_data.csv")
        return df
    except:
        # Generate new sample data if file doesn't exist
        # Sample project data
        projects = [
            "Cloud Migration",
            "Web App Development",
            "Mobile App",
            "Data Center Setup",
            "CRM Implementation",
            "Security Audit",
            "DevOps Pipeline",
            "IoT Platform",
            "AI Chatbot",
            "Network Upgrade"
        ]
        
        project_types = [
            "Infrastructure",
            "Software Development",
            "Software Development",
            "Infrastructure",
            "Software Development",
            "Consulting",
            "Infrastructure",
            "Software Development",
            "Research",
            "Maintenance"
        ]
        
        clients = [
            "Acme Corp",
            "TechGiant",
            "MediHealth",
            "FinancePro",
            "RetailMaster",
            "GovAgency",
            "StartupX",
            "EduTech",
            "ManufactureAll",
            "LogisticsPrime"
        ]
        
        # Generate data with varying risk levels
        data = []
        today = datetime.datetime.now()
        
        for i in range(len(projects)):
            # Generate realistic project data with varied risk factors
            delay_days = random.randint(0, 25)
            
            payment_options = ["On Time", "Late", "Missed"]
            payment_weights = [0.5, 0.3, 0.2]
            payment_status = random.choices(payment_options, payment_weights)[0]
            
            resignations = random.randint(0, 3)
            
            # Calculate base risk score
            risk_score = calculate_risk_score(delay_days, payment_status, resignations)
            
            # Add some randomness for variety
            start_date = today - datetime.timedelta(days=random.randint(30, 180))
            deadline = start_date + datetime.timedelta(days=random.randint(90, 365))
            
            # Budget and completion metrics
            budget = random.randint(50000, 500000)
            spent = budget * (random.uniform(0.5, 1.2))  # Some under, some over budget
            completion = min(100, max(10, random.randint(20, 100)))
            
            data.append({
                "project": projects[i],
                "project_type": project_types[i],
                "client": clients[i],
                "start_date": start_date.strftime("%Y-%m-%d"),
                "deadline": deadline.strftime("%Y-%m-%d"),
                "delay_days": delay_days,
                "payment_status": payment_status,
                "resignations": resignations,
                "risk_score": risk_score,
                "risk_level": get_risk_level(risk_score),
                "budget": budget,
                "spent": spent,
                "completion": completion,
                "team_size": random.randint(3, 12)
            })
        
        df = pd.DataFrame(data)
        df.to_csv("project_data.csv", index=False)
        return df

def load_sample_news_data():
    """Load or generate sample market news data"""
    try:
        # Try to load existing data
        with open("dummy.json", "r") as f:
            return json.load(f)
    except:
        # Generate new sample news data
        news = [
            {
                "title": "Tech Sector Faces Economic Slowdown",
                "content": "Analysts predict a significant slowdown in tech spending as companies tighten budgets amid economic uncertainty.",
                "date": (datetime.datetime.now() - datetime.timedelta(days=2)).strftime("%Y-%m-%d"),
                "source": "Tech Insights"
            },
            {
                "title": "Cloud Computing Demand Still Strong",
                "content": "Despite economic headwinds, cloud service providers report continued growth as businesses accelerate digital transformation.",
                "date": (datetime.datetime.now() - datetime.timedelta(days=4)).strftime("%Y-%m-%d"),
                "source": "Cloud Industry Review"
            },
            {
                "title": "Cybersecurity Skills Shortage Worsens",
                "content": "The gap between cybersecurity job openings and qualified professionals continues to widen, driving up hiring costs.",
                "date": (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
                "source": "Security Weekly"
            },
            {
                "title": "Supply Chain Issues Impact Hardware Delivery",
                "content": "Global supply chain disruptions continue to delay hardware deliveries for IT infrastructure projects worldwide.",
                "date": (datetime.datetime.now() - datetime.timedelta(days=5)).strftime("%Y-%m-%d"),
                "source": "Supply Chain Monitor"
            },
            {
                "title": "AI Development Costs on the Rise",
                "content": "Companies implementing AI solutions report higher than expected costs due to computing resources and talent acquisition.",
                "date": (datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d"),
                "source": "AI Industry Today"
            }
        ]
        
        # Save the data
        with open("news_feed.json", "w") as f:
            json.dump(news, f)
        
        return news

# Gemini chatbot functions
def initialize_gemini():
    """Initialize the Gemini model"""
    try:
        # Check if API key is in environment variables
        # api_key = os.getenv("GEMINI_API_KEY")
        api_key = st.secrets["GEMINI_API_KEY"]
        
        # If not found in environment, try to get from session state
        if not api_key and "gemini_api_key" in st.session_state:
            api_key = st.session_state.gemini_api_key
            
        if api_key:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')
            return model
        return None
    except Exception as e:
        st.error(f"Error initializing Gemini: {e}")
        return None

def query_gemini(model, query, project_data_context, chat_history=None):
    """Query the Gemini model with project context"""
    if not model:
        return "Please set up your Gemini API key first."
    
    try:
        # Create a prompt with context about the project data
        system_prompt = f"""
        You are an AI Project Risk Management Assistant. You help project managers understand risks and suggest mitigation strategies.
        
        Here's the current project data:
        {project_data_context}
        
        Please analyze this data when answering questions. Be specific, practical and concise in your answers.
        Focus on risk identification, assessment, and mitigation strategies.
        You do not take any other topic as discussion
        Stick to the project risk management
        Project risk analysis only
        """
        
        # If we have chat history, use it for context
        if chat_history and len(chat_history) > 0:
            history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history])
            system_prompt += f"\n\nRecent conversation history:\n{history_text}"
        
        # Full prompt
        full_prompt = f"{system_prompt}\n\nUser question: {query}"
        
        # Get response from Gemini
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error querying Gemini: {e}"


# Main Streamlit UI
def main():

    # Initialize Gemini model
    gemini_model = initialize_gemini()

    # Load sample data if live news does not works
    df_projects = load_sample_project_data()
    news_data = load_sample_news_data()
    
    # Calculate additional metrics
    df_projects['budget_variance'] = ((df_projects['spent'] / df_projects['budget']) - 1) * 100
    
    # Check if master news list is already in session state
    try:
        if 'master_news' not in st.session_state:
            # Create master news list for all projects
            with st.spinner("Fetching live data for all projects...It may take a few seconds."):
                st.session_state.master_news = create_master_news_list(df_projects, gemini_model)
    except:
        pass
        
    
    # Update risk scores with external factors (market risk)
    for idx, row in df_projects.iterrows():
        project_name = row["project"]
        
        # Get news for this project
        project_news = st.session_state.master_news.get(project_name, [])
        
        # Calculate average sentiment across project news
        if project_news:
            avg_sentiment = sum(n["sentiment"] for n in project_news) / len(project_news)
        else:
            avg_sentiment = 0.0  # Neutral if no news
        
        # Add market impact to risk score
        market_risk = market_impact_on_project(row["project_type"], avg_sentiment)
        
        # Update risk score with market impact
        updated_risk_score = min(100, row["risk_score"] + market_risk)
        df_projects.at[idx, "market_risk"] = market_risk
        df_projects.at[idx, "final_risk_score"] = updated_risk_score
        df_projects.at[idx, "final_risk_level"] = get_risk_level(updated_risk_score)
    
    # Sidebar
    with st.sidebar:
        st.title("🛠️ AI Project Risk Manager")
        st.caption("Advanced risk analysis and monitoring")
        
        # Navigation
        st.subheader("Navigation")
        page = st.radio("", ["Dashboard", "Project Details", "Market Analysis", "AI Chatbot"], 
                         format_func=lambda x: f"📊 {x}" if x == "Dashboard" 
                                     else f"📋 {x}" if x == "Project Details"
                                     else f"📰 {x}" if x == "Market Analysis"
                                     else f"🤖 {x}")
        
        # Project selector (available in all sections)
        st.divider()
        st.subheader("Project Selection")
        selected_project = st.selectbox("Choose Project", df_projects["project"].tolist())
        st.session_state.selected_project = selected_project
        
        # # Gemini API key input
        # st.divider()
        # st.subheader("🔑 Gemini API Setup")
        # api_key = st.text_input("Gemini API Key", 
        #                        value=st.session_state.get("gemini_api_key", ""), 
        #                        type="password",
        #                        help="Enter your Google Gemini API key to enable the AI chatbot")
        
        # if api_key:
        #     st.session_state.gemini_api_key = api_key
        #     if not gemini_model:
        #         gemini_model = initialize_gemini()
        #         if gemini_model:
        #             st.success("Gemini API connected successfully!")
                    
    # Get selected project data
    project_data = df_projects[df_projects["project"] == st.session_state.selected_project].iloc[0].to_dict()

    
    # Main content area
    if page == "Dashboard":
        # Overview Dashboard
        st.title("📊 Project Risk Dashboard")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("High Risk Projects", len(df_projects[df_projects["final_risk_level"] == "High"]))
        with col2:
            st.metric("Medium Risk Projects", len(df_projects[df_projects["final_risk_level"] == "Medium"]))
        with col3:
            st.metric("Low Risk Projects", len(df_projects[df_projects["final_risk_level"] == "Low"]))
        with col4:
            avg_risk = df_projects["final_risk_score"].mean()
            st.metric("Avg. Risk Score", f"{avg_risk:.1f}/100")
        
        # Project risk chart
        st.subheader("Project Risk Overview")
        
        # Sort by risk score for better visualization
        df_sorted = df_projects.sort_values("final_risk_score", ascending=False)
        
        # Create bar chart
        fig = px.bar(df_sorted, 
                     x="project", 
                     y="final_risk_score",
                     color="final_risk_level",
                     color_discrete_map={"High": "#FF4B4B", "Medium": "#FFA500", "Low": "#00CC96"},
                     labels={"project": "Project", "final_risk_score": "Risk Score", "final_risk_level": "Risk Level"},
                     title="Project Risk Scores",
                     text="final_risk_score")
        
        fig.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk factors breakdown
        st.subheader("Risk Factor Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Schedule risks
            schedule_risk_data = df_projects[["project", "delay_days"]].sort_values("delay_days", ascending=False).head(5)
            fig = px.bar(schedule_risk_data, 
                         x="project", 
                         y="delay_days",
                         title="Top Schedule Risks (Delay Days)",
                         color="delay_days",
                         color_continuous_scale=[(0, "#00CC96"), (0.5, "#FFA500"), (1, "#FF4B4B")])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Resource risks
            resource_risk_data = df_projects[["project", "resignations"]].sort_values("resignations", ascending=False).head(5)
            fig = px.bar(resource_risk_data, 
                         x="project", 
                         y="resignations",
                         title="Top Resource Risks (Resignations)",
                         color="resignations",
                         color_continuous_scale=[(0, "#00CC96"), (0.5, "#FFA500"), (1, "#FF4B4B")])
            st.plotly_chart(fig, use_container_width=True)
            
        # Budget variance
        st.subheader("Budget Performance")
        budget_data = df_projects[["project", "budget_variance"]].sort_values("budget_variance", ascending=False)
        
        fig = px.bar(budget_data,
                    x="project",
                    y="budget_variance",
                    title="Budget Variance (%)",
                    color="budget_variance",
                    color_continuous_scale=[(0, "#00CC96"), (0.5, "#FFA500"), (1, "#FF4B4B")])
        
        fig.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Project Details":
        # Project Details Page
        st.title(f"📋 {project_data['project']} - Risk Analysis")
        st.caption(f"Client: {project_data['client']} | Type: {project_data['project_type']}")
        
        # Risk score and summary
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Risk score gauge
            gauge_fig = create_risk_gauge(project_data["final_risk_score"], "Project Risk Score")
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Market risk impact
            st.metric("Market Risk Impact", f"+{project_data['market_risk']:.1f} points")
            
        with col2:
            # Risk summary
            st.subheader("Risk Assessment")
            st.markdown(project_risk_summary(project_data))
        
        # Detailed metrics
        st.divider()
        st.subheader("Project Metrics")
        
        # Display metrics in columns
        metrics = create_project_metrics_table(project_data)
        col1, col2, col3, col4, col5 = st.columns(5)
        
        cols = [col1, col2, col3, col4, col5]
        for i, (key, value) in enumerate(metrics.items()):
            cols[i % 5].metric(key, value)
        
        # Risk factors chart
        st.divider()
        factors_fig = create_risk_factors_chart(project_data)
        st.plotly_chart(factors_fig, use_container_width=True)
        
        # Project timeline
        st.divider()
        st.subheader("Project Timeline")
        
        # Calculate timeline metrics
        start_date = datetime.datetime.strptime(project_data['start_date'], "%Y-%m-%d")
        deadline = datetime.datetime.strptime(project_data['deadline'], "%Y-%m-%d")
        today = datetime.datetime.now()
        
        total_days = (deadline - start_date).days
        elapsed_days = (today - start_date).days
        remaining_days = (deadline - today).days
        
        # Expected vs actual progress
        expected_progress = min(100, max(0, (elapsed_days / total_days) * 100))
        actual_progress = project_data['completion']
        
        # Timeline visualization
        timeline_data = pd.DataFrame([
            {"Category": "Expected Progress", "Value": expected_progress},
            {"Category": "Actual Progress", "Value": actual_progress}
        ])
        
        fig = px.bar(timeline_data, 
                     x="Category", 
                     y="Value",
                     color="Category",
                     color_discrete_map={"Expected Progress": "#4B9CD3", "Actual Progress": "#FFB347"},
                     labels={"Value": "Completion %"},
                     title="Project Progress (Expected vs. Actual)")
        
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
        
        # Project status summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Duration", f"{total_days} days")
        with col2:
            st.metric("Days Elapsed", f"{elapsed_days} days")
        with col3:
            st.metric("Days Remaining", f"{remaining_days} days",
                     delta=f"{remaining_days - project_data['delay_days']} adjusted")
    
    elif page == "Market Analysis":
        # Market Analysis Page (modified)
        st.title("📰 Market Risk Analysis")
        
        # News feed - Show news for selected project
        project_news = st.session_state.master_news.get(st.session_state.selected_project, [])
        
        # Add a tab for All News and Project-specific News
        tab1, tab2 = st.tabs(["Project News", "All Market News"])
        
        with tab1:
            st.subheader(f"News Related to: {st.session_state.selected_project}")
            
            if not project_news:
                st.info(f"No news found for {st.session_state.selected_project}. Try selecting another project.")
            else:
                # Calculate average sentiment for this project
                avg_sentiment = sum(n["sentiment"] for n in project_news) / len(project_news)
                sentiment_status = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
                sentiment_color = "#00CC96" if avg_sentiment > 0.05 else "#FF4B4B" if avg_sentiment < -0.05 else "#FFA500"
                
                # Display project sentiment summary
                st.metric(
                    "Overall Market Sentiment", 
                    sentiment_status, 
                    f"{avg_sentiment:.2f}", 
                    delta_color="normal" if sentiment_status == "Positive" else "inverse"
                )
                
                # Display project news
                for news in project_news:
                    formatted_date = format_published_date(news.get("publishedAt", "Unknown date"))
                    
                    with st.expander(f"{news.get('title', 'No title')} ({formatted_date})"):
                        st.write(news.get("content", "No content available"))
                        st.caption(f"Source: {news.get('source', {}).get('name', 'Unknown source')}")
                        st.metric(
                            "Sentiment", 
                            news["sentiment_label"], 
                            f"{news['sentiment']:.2f}"
                        )
        
        with tab2:
            st.subheader("All Market News")
            
            # Flatten all news articles from all projects
            all_news = []
            for news_list in st.session_state.master_news.values():
                all_news.extend(news_list)
            
            # Remove duplicates based on title
            unique_news = {}
            for news in all_news:
                if news.get("title") not in unique_news:
                    unique_news[news.get("title")] = news
            
            all_news = list(unique_news.values())
            
            # Sort by date (newest first)
            all_news = sorted(
                all_news, 
                key=lambda x: x.get("publishedAt", ""), 
                reverse=True
            )
            
            for news in all_news:
                formatted_date = format_published_date(news.get("publishedAt", "Unknown date"))
                
                with st.expander(f"{news.get('title', 'No title')} ({formatted_date})"):
                    st.write(news.get("content", "No content available"))
                    st.caption(f"Source: {news.get('source', {}).get('name', 'Unknown source')}")
                    st.metric(
                        "Sentiment", 
                        news["sentiment_label"], 
                        f"{news['sentiment']:.2f}"
                    )
        
        # Market impact on projects (rest of the code remains the same)
        st.divider()
        st.subheader("Market Impact on Projects")
        
        # Prepare data
        market_impact_data = df_projects[["project", "project_type", "risk_score", "market_risk", "final_risk_score"]]
        market_impact_data = market_impact_data.sort_values("market_risk", ascending=False)
        
        # Highlight selected project
        market_impact_data["is_selected"] = market_impact_data["project"] == st.session_state.selected_project
        
        # Market impact chart
        fig = px.bar(market_impact_data,
                     x="project",
                     y=["risk_score", "market_risk"],
                     labels={"value": "Risk Score", "variable": "Risk Component"},
                     title="Base Risk vs. Market-Added Risk",
                     color_discrete_map={"risk_score": "#4B9CD3", "market_risk": "#FF4B4B"})
        
        # Highlight selected project with a box
        if st.session_state.selected_project:
            selected_idx = market_impact_data[market_impact_data["project"] == st.session_state.selected_project].index[0]
            fig.add_shape(
                type="rect",
                xref="x",
                yref="paper",
                x0=selected_idx-0.4,
                y0=0,
                x1=selected_idx+0.4,
                y1=1,
                line=dict(color="Gold", width=3),
                fillcolor="rgba(255, 215, 0, 0.1)"
            )
        
        fig.update_layout(legend=dict(
            title="Risk Component",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Project type sensitivity analysis
        st.divider()
        st.subheader("Project Type Sensitivity to Market")
        
        project_type_impact = market_impact_data.groupby("project_type").agg({
            "market_risk": "mean",
            "project": "count"
        }).reset_index().rename(columns={"project": "count"})
        
        # Highlight the selected project's type
        project_type_impact["is_selected"] = project_type_impact["project_type"] == project_data["project_type"]
        
        fig = px.bar(project_type_impact,
                    x="project_type",
                    y="market_risk",
                    title="Average Market Risk Impact by Project Type",
                    color="market_risk",
                    color_continuous_scale=[(0, "#00CC96"), (0.5, "#FFA500"), (1, "#FF4B4B")],
                    text="count")
        
        # Highlight selected project type
        for i, row in project_type_impact.iterrows():
            if row["project_type"] == project_data["project_type"]:
                fig.add_shape(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=i-0.4,
                    y0=0,
                    x1=i+0.4,
                    y1=1,
                    line=dict(color="Gold", width=3),
                    fillcolor="rgba(255, 215, 0, 0.1)"
                )
        
        fig.update_traces(texttemplate='%{text} projects', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Selected project market impact
        st.divider()
        st.subheader(f"Market Impact on {project_data['project']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Base vs final risk
            labels = ['Base Risk', 'Market Risk']
            values = [project_data['risk_score'], project_data['market_risk']]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.4,
                marker_colors=['#4B9CD3', '#FF4B4B']
            )])
            
            fig.update_layout(title_text="Risk Composition")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Market risk explanation
            st.subheader("Market Risk Analysis")
            st.write(f"Project type: **{project_data['project_type']}**")
            
            # Calculate average sentiment for the selected project
            project_news = st.session_state.master_news.get(st.session_state.selected_project, [])
            if project_news:
                avg_sentiment = sum(n["sentiment"] for n in project_news) / len(project_news)
                st.write(f"Current market sentiment: **{avg_sentiment:.2f}**")
                
                if avg_sentiment < -0.1:
                    st.error("Negative market conditions increase project risk")
                elif avg_sentiment > 0.1:
                    st.success("Positive market conditions mitigate project risk")
                else:
                    st.info("Neutral market conditions with minimal impact")
            else:
                st.write("No market sentiment data available for this project")
                
            # Market risk breakdown
            st.write(f"Market risk contribution: **+{project_data['market_risk']:.1f} points** to risk score")
            st.write(f"Final risk score: **{project_data['final_risk_score']:.1f}** (Base: {project_data['risk_score']:.1f})")

            
    elif page == "AI Chatbot":
        # Chatbot page
        st.title("🤖 AI Risk Management Assistant")
        
        # API key check
        if not gemini_model:
            st.warning("Please enter your Gemini API key in the sidebar to use the AI assistant.")
            st.info("You can get a Gemini API key from https://ai.google.dev/")
        else:
            # Prepare project context for the AI
            project_context = f"""
            Current Project: {project_data['project']} (Client: {project_data['client']})
            Project Type: {project_data['project_type']}
            Risk Score: {project_data['final_risk_score']:.1f}/100 ({project_data['final_risk_level']} Risk)
            Key Metrics:
            - Schedule: {project_data['delay_days']} days delay
            - Payment Status: {project_data['payment_status']}
            - Team Resignations: {project_data['resignations']}
            - Budget Variance: {((project_data['spent'] / project_data['budget']) - 1) * 100:.1f}%
            - Completion: {project_data['completion']}%
            
            All Projects Summary:
            - High Risk Projects: {len(df_projects[df_projects["final_risk_level"] == "High"])}
            - Medium Risk Projects: {len(df_projects[df_projects["final_risk_level"] == "Medium"])}
            - Low Risk Projects: {len(df_projects[df_projects["final_risk_level"] == "Low"])}
            """
            
            # Display chat interface
            st.subheader(f"Ask about risks for: {project_data['project']}")
            
            # Display chat history
            for question, answer in st.session_state.chat_history:
                st.info(f"You: {question}")
                st.success(f"AI Assistant: {answer}")
            
            # Chat input
            user_query = st.text_input(
                "Ask a question about project risks:", 
                placeholder="E.g., What's the biggest risk for this project? Or, How can we mitigate the schedule delay?"
            )
            
            # Sample questions
            with st.expander("Sample questions to ask"):
                st.markdown("""
                - What are the main risk factors for this project?
                - How is the market affecting our project risk?
                - What mitigation strategies do you recommend?
                - How can we address the resource issues?
                - Which projects in our portfolio need immediate attention?
                - How can we improve our payment situation?
                - What's the trend of our schedule performance?
                - Compare the risk level with other similar projects
                """)
            
            # Process query
            if user_query:
                with st.spinner("Analyzing project data and generating response..."):
                    response = query_gemini(
                        gemini_model, 
                        user_query, 
                        project_context, 
                        st.session_state.chat_history
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append((user_query, response))
                    
                    # Display latest response
                    st.info(f"You: {user_query}")
                    st.success(f"AI Assistant: {response}")
            
            # Clear chat button
            if st.button("Clear Chat History") and st.session_state.chat_history:
                st.session_state.chat_history = []
                st.experimental_rerun()

if __name__ == "__main__":
    main()
