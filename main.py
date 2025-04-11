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

# Helper functions for risk scoring
def calculate_risk_score(delay_days, payment_status, resignations):
    """Calculate risk score based on project metrics"""
    score = 0
    
    # Schedule delay impact
    if delay_days <= 5:
        score += delay_days * 2  # 2 points per day for small delays
    elif 5 < delay_days <= 15:
        score += 10 + (delay_days - 5) * 3  # Higher penalty for medium delays
    else:
        score += 40  # Maximum penalty for severe delays
    
    # Payment issues
    if payment_status == "Late":
        score += 15
    elif payment_status == "Missed":
        score += 30
    
    # Resource issues - resignations
    if resignations == 1:
        score += 15
    elif resignations > 1:
        score += 15 + (resignations - 1) * 10  # Each additional resignation adds 10 points
        
    return min(score, 100)  # Cap at 100

def get_risk_level(score):
    """Convert numerical score to categorical risk level"""
    if score >= HIGH_RISK_THRESHOLD:
        return "High"
    elif score >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    else:
        return "Low"

def get_risk_color(score):
    """Get color for risk visualization"""
    if score >= HIGH_RISK_THRESHOLD:
        return "#FF4B4B"  # Red
    elif score >= MEDIUM_RISK_THRESHOLD:
        return "#FFA500"  # Orange
    else:
        return "#00CC96"  # Green

# Market Analysis functions
def analyze_news_sentiment(news_text):
    """Analyze sentiment of news text"""
    blob = TextBlob(news_text)
    return blob.sentiment.polarity  # -1 to 1 (negative to positive)

def market_impact_on_project(project_type, sentiment_score):
    """Calculate market impact on project based on sentiment and project type"""
    # Different project types have different sensitivity to market sentiment
    sensitivity = {
        "Software Development": 0.4,
        "Infrastructure": 0.7,
        "Consulting": 0.6,
        "Maintenance": 0.3,
        "Research": 0.5
    }
    
    project_sensitivity = sensitivity.get(project_type, 0.5)
    # Convert sentiment (-1 to 1) to risk impact (0 to 30)
    # Negative sentiment increases risk, positive sentiment decreases risk
    impact = ((-sentiment_score) * project_sensitivity) * 30
    
    # Ensure impact is between 0 and 30
    return max(0, min(30, impact))

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
        with open("news_feed.json", "r") as f:
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
        api_key = os.getenv("GEMINI_API_KEY")
        
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

# Custom functions for reporting and dashboards
def create_risk_gauge(score, title="Overall Risk Score"):
    """Create a risk gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': get_risk_color(score)},
            'steps': [
                {'range': [0, MEDIUM_RISK_THRESHOLD], 'color': "rgba(0, 204, 150, 0.3)"},
                {'range': [MEDIUM_RISK_THRESHOLD, HIGH_RISK_THRESHOLD], 'color': "rgba(255, 165, 0, 0.3)"},
                {'range': [HIGH_RISK_THRESHOLD, 100], 'color': "rgba(255, 75, 75, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': HIGH_RISK_THRESHOLD
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_risk_factors_chart(project_data):
    """Create a risk factor breakdown chart"""
    # Calculate risk components
    delay_risk = min(40, project_data['delay_days'] * 2 if project_data['delay_days'] <= 5 
                    else 10 + (project_data['delay_days'] - 5) * 3 if project_data['delay_days'] <= 15
                    else 40)
    
    payment_risk = 0
    if project_data['payment_status'] == "Late":
        payment_risk = 15
    elif project_data['payment_status'] == "Missed":
        payment_risk = 30
        
    resource_risk = 15 if project_data['resignations'] == 1 else 15 + (project_data['resignations'] - 1) * 10 if project_data['resignations'] > 1 else 0
    resource_risk = min(30, resource_risk)
    
    # Create bar chart
    categories = ['Schedule Risk', 'Payment Risk', 'Resource Risk']
    values = [delay_risk, payment_risk, resource_risk]
    colors = ["#FF9966", "#6699FF", "#99FF99"]
    
    fig = go.Figure(go.Bar(
        x=categories,
        y=values,
        text=values,
        textposition='auto',
        marker_color=colors
    ))
    
    fig.update_layout(
        title="Risk Factor Breakdown",
        yaxis=dict(title="Risk Impact", range=[0, 50]),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_project_metrics_table(project_data):
    """Create a table of key project metrics"""
    metrics = {
        "Start Date": project_data["start_date"],
        "Deadline": project_data["deadline"],
        "Budget": f"${project_data['budget']:,}",
        "Spent": f"${project_data['spent']:.2f}",
        "Budget Variance": f"{((project_data['spent'] / project_data['budget']) - 1) * 100:.1f}%",
        "Completion": f"{project_data['completion']}%",
        "Team Size": f"{project_data['team_size']} members",
        "Delay": f"{project_data['delay_days']} days",
        "Payment Status": project_data["payment_status"],
        "Resignations": f"{project_data['resignations']} people"
    }
    
    return metrics

def project_risk_summary(project_data):
    """Generate a risk summary for a project"""
    risk_level = project_data["risk_level"]
    risk_score = project_data["risk_score"]
    
    summary = f"**Risk Level: {risk_level}** (Score: {risk_score}/100)\n\n"
    
    # Add specific risk factors
    if project_data["delay_days"] > 10:
        summary += "• **Critical Schedule Risk**: Project is significantly behind schedule.\n"
    elif project_data["delay_days"] > 5:
        summary += "• **Moderate Schedule Risk**: Project is somewhat behind schedule.\n"
    
    if project_data["payment_status"] == "Missed":
        summary += "• **Critical Payment Risk**: Client has missed payments.\n"
    elif project_data["payment_status"] == "Late":
        summary += "• **Payment Concern**: Client payments are delayed.\n"
    
    if project_data["resignations"] > 1:
        summary += f"• **Critical Resource Risk**: Multiple team members ({project_data['resignations']}) have resigned.\n"
    elif project_data["resignations"] == 1:
        summary += "• **Resource Concern**: One team member has resigned.\n"
    
    if project_data["spent"] > project_data["budget"]:
        overspent = ((project_data["spent"] / project_data["budget"]) - 1) * 100
        summary += f"• **Budget Risk**: Project is {overspent:.1f}% over budget.\n"
    
    # Suggest mitigation strategies
    summary += "\n**Recommended Actions**:\n"
    
    if project_data["delay_days"] > 5:
        summary += "• Review project timeline and consider scope adjustments\n"
        
    if project_data["payment_status"] != "On Time":
        summary += "• Escalate payment issues to account management\n"
        
    if project_data["resignations"] > 0:
        summary += "• Accelerate hiring process and redistribute workload\n"
        
    if project_data["spent"] > project_data["budget"]:
        summary += "• Conduct budget review and implement cost controls\n"
    
    return summary

# Main Streamlit UI
def main():
    # Load sample data
    df_projects = load_sample_project_data()
    news_data = load_sample_news_data()
    
    # Calculate additional metrics
    df_projects['budget_variance'] = ((df_projects['spent'] / df_projects['budget']) - 1) * 100
    
    # Add market risk based on news sentiment
    for i, news in enumerate(news_data):
        news["sentiment"] = analyze_news_sentiment(news["content"])
        news["sentiment_label"] = "Positive" if news["sentiment"] > 0.05 else "Negative" if news["sentiment"] < -0.05 else "Neutral"
    
    # Update risk scores with external factors (market risk)
    for idx, row in df_projects.iterrows():
        # Calculate average sentiment across all news
        avg_sentiment = sum(n["sentiment"] for n in news_data) / len(news_data)
        
        # Add market impact to risk score
        market_risk = market_impact_on_project(row["project_type"], avg_sentiment)
        
        # Update risk score with market impact
        updated_risk_score = min(100, row["risk_score"] + market_risk)
        df_projects.at[idx, "market_risk"] = market_risk
        df_projects.at[idx, "final_risk_score"] = updated_risk_score
        df_projects.at[idx, "final_risk_level"] = get_risk_level(updated_risk_score)
    
    # Initialize Gemini model
    gemini_model = initialize_gemini()
    
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
        
        # Gemini API key input
        st.divider()
        st.subheader("🔑 Gemini API Setup")
        api_key = st.text_input("Gemini API Key", 
                               value=st.session_state.get("gemini_api_key", ""), 
                               type="password",
                               help="Enter your Google Gemini API key to enable the AI chatbot")
        
        if api_key:
            st.session_state.gemini_api_key = api_key
            if not gemini_model:
                gemini_model = initialize_gemini()
                if gemini_model:
                    st.success("Gemini API connected successfully!")
                    
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
        # Market Analysis Page
        st.title("📰 Market Risk Analysis")
        
        # News feed
        st.subheader("Recent Market News")
        
        for news in news_data:
            sentiment_color = "#00CC96" if news["sentiment"] > 0.05 else "#FF4B4B" if news["sentiment"] < -0.05 else "#FFA500"
            
            with st.expander(f"{news['title']} ({news['date']})"):
                st.write(news["content"])
                st.caption(f"Source: {news['source']}")
                st.metric("Sentiment", news["sentiment_label"], f"{news['sentiment']:.2f}")
        
        # Market impact on projects
        st.divider()
        st.subheader("Market Impact on Projects")
        
        # Prepare data
        market_impact_data = df_projects[["project", "project_type", "risk_score", "market_risk", "final_risk_score"]]
        market_impact_data = market_impact_data.sort_values("market_risk", ascending=False)
        
        # Market impact chart
        fig = px.bar(market_impact_data,
                     x="project",
                     y=["risk_score", "market_risk"],
                     labels={"value": "Risk Score", "variable": "Risk Component"},
                     title="Base Risk vs. Market-Added Risk",
                     color_discrete_map={"risk_score": "#4B9CD3", "market_risk": "#FF4B4B"})
        
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
        
        fig = px.bar(project_type_impact,
                    x="project_type",
                    y="market_risk",
                    title="Average Market Risk Impact by Project Type",
                    color="market_risk",
                    color_continuous_scale=[(0, "#00CC96"), (0.5, "#FFA500"), (1, "#FF4B4B")],
                    text="count")
        
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
            # Market risk explanation
            st.subheader("Market Risk Analysis")
            st.write(f"Project type: **{project_data['project_type']}**")
            
            # Calculate average sentiment
            avg_sentiment = sum(n["sentiment"] for n in news_data) / len(news_data)
            st.write(f"Current market sentiment: **{avg_sentiment:.2f}**")
            
            if avg_sentiment < -0.1:
                st.error("Negative market conditions increase project risk")
            elif avg_sentiment > 0.1:
                st.success("Positive market conditions mitigate project risk")
            else:
                st.info("Neutral market conditions with minimal impact")
                
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