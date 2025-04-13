import datetime
import plotly.graph_objects as go
from utils.utils import fetch_news, analyze_news_sentiment, generate_query
import streamlit as st
# Constants
HIGH_RISK_THRESHOLD = 70
MEDIUM_RISK_THRESHOLD = 40


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

def format_published_date(published_at):
    """Convert ISO 8601 date to a user-friendly format."""
    try:
        date_obj = datetime.datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
        return date_obj.strftime("%B %d, %Y at %I:%M %p")
    except ValueError:
        return published_at  # Return as-is if parsing fails

