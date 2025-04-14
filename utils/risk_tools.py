import datetime
import plotly.graph_objects as go
from utils.utils import fetch_news, analyze_news_sentiment, generate_query
import streamlit as st
import requests
import json
from datetime import datetime, timedelta

# Jira Integration Utility
class JiraConnector:
    """Class to handle Jira integration and API calls"""
    
    def __init__(self, base_url, email, api_token):
        """
        Initialize Jira connector
        
        Args:
            base_url (str): Jira instance URL (e.g., 'https://ezioauditoredefirenze132003.atlassian.net')
            email (str): Jira account email
            api_token (str): Jira API token
        """
        self.base_url = base_url
        self.auth = (email, api_token)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    
    def test_connection(self):
        """Test the Jira connection"""
        try:
            response = requests.get(
                f"{self.base_url}/rest/api/3/myself",
                headers=self.headers,
                auth=self.auth
            )
            if response.status_code == 200:
                return True, "Connection successful"
            else:
                return False, f"Connection failed with status {response.status_code}: {response.text}"
        except Exception as e:
            return False, f"Connection error: {str(e)}"
    
    def get_project_by_key(self, project_key):
        """Get project details by project key"""
        try:
            response = requests.get(
                f"{self.base_url}/rest/api/3/project/{project_key}",
                headers=self.headers,
                auth=self.auth
            )
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            print(f"Error fetching project: {str(e)}")
            return None
    
    def get_project_issues(self, project_key, max_results=100):
        """Get issues for a specific project"""
        try:
            jql_query = f"project = {project_key} ORDER BY created DESC"
            response = requests.get(
                f"{self.base_url}/rest/api/3/search",
                headers=self.headers,
                auth=self.auth,
                params={
                    "jql": jql_query,
                    "maxResults": max_results
                }
            )
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            print(f"Error fetching issues: {str(e)}")
            return None
    
    def get_project_sprints(self, board_id):
        """Get sprint information for a project board"""
        try:
            response = requests.get(
                f"{self.base_url}/rest/agile/1.0/board/{board_id}/sprint",
                headers=self.headers,
                auth=self.auth,
                params={"state": "active,closed"}
            )
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            print(f"Error fetching sprints: {str(e)}")
            return None
    
    def get_project_boards(self, project_key):
        """Get boards associated with a project"""
        try:
            response = requests.get(
                f"{self.base_url}/rest/agile/1.0/board",
                headers=self.headers,
                auth=self.auth,
                params={"projectKeyOrId": project_key}
            )
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            print(f"Error fetching boards: {str(e)}")
            return None

def fetch_jira_project_data(jira_connector, project_key):
    """
    Fetch and process Jira project data for risk analysis
    
    Args:
        jira_connector: JiraConnector instance
        project_key: Jira project key
        
    Returns:
        dict: Dictionary with project metrics that can be used for risk analysis
    """
    # Get basic project information
    project = jira_connector.get_project_by_key(project_key)
    if not project:
        return None
    
    # Get issues for risk analysis
    issues_data = jira_connector.get_project_issues(project_key)
    if not issues_data:
        return None
    
    issues = issues_data.get("issues", [])
    
    # Get board and sprint information
    boards = jira_connector.get_project_boards(project_key)
    sprint_data = None
    if boards and boards.get("values"):
        board_id = boards.get("values")[0].get("id")
        sprint_data = jira_connector.get_project_sprints(board_id)
    
    # Process and return data for risk analysis
    # This can be expanded to extract specific metrics needed for risk calculation
    return {
        "project": project,
        "issues": issues,
        "sprints": sprint_data.get("values", []) if sprint_data else []
    }

def extract_risk_metrics_from_jira(jira_data):
    """
    Extract risk-related metrics from Jira data
    
    Args:
        jira_data: Dictionary containing Jira project data
        
    Returns:
        dict: Dictionary with risk metrics that can be used with existing risk analysis
    """
    metrics = {}
    
    if not jira_data:
        return metrics
    
    project = jira_data.get("project", {})
    issues = jira_data.get("issues", [])
    sprints = jira_data.get("sprints", [])
    
    # Extract project name and key
    metrics["project_name"] = project.get("name", "")
    metrics["project_key"] = project.get("key", "")
    
    # Calculate delay days based on due dates
    overdue_issues = 0
    delay_days = 0
    today = datetime.now()
    
    for issue in issues:
        fields = issue.get("fields", {})
        due_date_str = fields.get("duedate")
        if due_date_str:
            try:
                due_date = datetime.strptime(due_date_str, "%Y-%m-%d")
                if due_date < today and fields.get("status", {}).get("name") != "Done":
                    overdue_issues += 1
                    issue_delay = (today - due_date).days
                    delay_days = max(delay_days, issue_delay)
            except ValueError:
                pass
    
    metrics["delay_days"] = delay_days
    
    # Calculate completion percentage
    total_issues = len(issues)
    completed_issues = sum(1 for issue in issues if issue.get("fields", {}).get("status", {}).get("name") == "Done")
    
    if total_issues > 0:
        completion_percentage = (completed_issues / total_issues) * 100
    else:
        completion_percentage = 0
    
    metrics["completion"] = round(completion_percentage)
    
    # Count team members and potential resignations (inactive members)
    assignees = set()
    for issue in issues:
        fields = issue.get("fields", {})
        assignee = fields.get("assignee")
        if assignee:
            assignees.add(assignee.get("key"))
    
    metrics["team_size"] = len(assignees)
    
    # This is a simplified proxy for resignations - would need more data in real implementation
    metrics["resignations"] = max(0, round(len(assignees) * 0.1))  # Placeholder estimate
    
    # Payment status - simplified proxy based on project health
    if delay_days > 15:
        metrics["payment_status"] = "Missed"
    elif delay_days > 7:
        metrics["payment_status"] = "Late"
    else:
        metrics["payment_status"] = "On Time"
    
    return metrics

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

import datetime  # Added import statement

def format_published_date(published_at):
    """Convert ISO 8601 date to a user-friendly format.
    
    Args:
        published_at (str): Date string in ISO 8601 format (e.g., '2023-01-15T14:30:00Z')
        
    Returns:
        str: Formatted date (e.g., 'January 15, 2023 at 02:30 PM') or original string if parsing fails
    """
    try:
        # Handle potential missing timezone indicator
        if published_at.endswith('Z'):
            date_obj = datetime.datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
        else:
            # Handle alternative formats if needed
            date_obj = datetime.datetime.fromisoformat(published_at)
            
        # Convert to local timezone (optional)
        # date_obj = date_obj.replace(tzinfo=datetime.timezone.utc).astimezone()
        
        return date_obj.strftime("%B %d, %Y at %I:%M %p")
    except (ValueError, TypeError):
        return published_at  # Return original if parsing fails
