# AI Project Risk Management System

## Overview

The AI Project Risk Management System is a powerful Streamlit-based application designed to help project managers monitor, analyze, and mitigate risks across their project portfolio. The system combines data visualization, real-time risk assessment, market analysis, and AI-powered insights to provide a comprehensive risk management solution.

## Features

- **Interactive Dashboard**: Real-time visualization of project risk scores, metrics, and status
- **Detailed Project Analysis**: In-depth examination of individual project risks and performance
- **Market Impact Analysis**: Integration of market news and its impact on project risk profiles
- **AI-Powered Chatbot**: Intelligent assistant for risk analysis and mitigation recommendations
- **Dynamic Data Visualization**: Rich graphical representations of risk metrics and factors

## System Architecture

The application follows a modular architecture with several key components:

```
AI Project Risk Management System
├── Main Application (app.py)
├── Utility Modules
│   ├── utils.py (Core utilities)
│   ├── risk_tools.py (Risk calculation functions)
├── LLM Integration
│   ├── Gemini API
│   ├── Langchain & LangGraph (workflow orchestration)
│   ├── Open-source LLMs (Llama, Mistral)
├── Vector Database
│   ├── ChromaDB (document storage and retrieval)
├── External Data Sources
│   ├── Project data
│   ├── Market news
```

## Core Functions

### Data Management

1. **`load_sample_project_data()`**: Loads existing project data or generates realistic sample data with risk factors.
   - Creates a dataset with project details, timelines, budgets, and risk metrics
   - Calculates initial risk scores based on delay days, payment status, and team resignations

2. **`load_sample_news_data()`**: Loads or generates market news data relevant to projects.
   - Creates structured news data with titles, content, dates, and sources

3. **`generate_dynamic_dashboard_values()`**: Generates updated risk scores and dashboard metrics.
   - Simulates real-time risk assessment with random variations
   - Incorporates market impact factors into risk calculations

4. **`project_scraping()`**: Real-time data collection with progress tracking.
   - Creates a visual representation of data gathering process
   - Updates project metrics with newly "scraped" information

### AI Integration

1. **`initialize_gemini()`**: Initializes the Google Gemini model for AI-powered analysis.
   - Securely manages API keys
   - Sets up the Gemini model for generating insights

2. **`query_gemini()`**: Queries the Gemini model with project context for personalized risk insights.
   - Constructs contextualized prompts with project data
   - Maintains conversation history for coherent interactions

3. **LangChain & LangGraph Integration**: Implements advanced workflow orchestration (imported from utils).
   - `initialize_langchain_pipeline`: Sets up LangChain processing pipeline
   - `create_langgraph_workflow`: Creates directed graph workflows for complex reasoning
   - `execute_langgraph_workflow`: Runs workflows with proper error handling

4. **Open-Source LLM Support**: Provides alternative model options for organizations with privacy constraints.
   - Functions to initialize and generate with Llama and Mistral models
   - Supports local deployment and HuggingFace model loading

### Risk Analysis

1. **`calculate_risk_score()`**: Calculates a numerical risk score based on multiple factors.
   - Considers schedule delays, payment issues, and team stability
   - Applies weighted scoring methodology

2. **`get_risk_level()`**: Converts numerical risk scores to categorical levels (High, Medium, Low).

3. **`project_risk_summary()`**: Generates comprehensive risk analysis for individual projects.

4. **`market_impact_on_project()`**: Calculates how market conditions affect specific project types.
   - Uses sentiment analysis of news to determine market impact
   - Adjusts risk scores based on project type sensitivity to market conditions

### Vector Database Operations

1. **ChromaDB Integration**: Enables semantic search and document retrieval.
   - `setup_chromadb_collection`: Creates and configures ChromaDB collections
   - `store_in_chromadb`: Indexes documents in vector database
   - `query_chromadb`: Performs semantic searches against stored documents

2. **Embedding Functions**: Creates and utilizes embeddings for semantic operations.
   - `create_embeddings`: Generates vector embeddings for text data
   - `similarity_search`: Finds semantically similar content
   - `load_sentence_transformer`: Initializes embedding models

## Implementation Details

### Error Mitigation through Callbacks

One of the most powerful features of this system is its robust error handling through callback mechanisms. This approach provides several benefits:

1. **Graceful Error Handling**: The system uses try-except blocks with callbacks to handle failures without crashing:

```python
try:
    # Initialize Gemini model
    gemini_model = initialize_gemini()
except Exception as e:
    st.error(f"Error initializing Gemini: {e}")
```

2. **Progress Tracking with Feedback**: For long-running operations, the system uses progress callbacks:

```python
def simulate_project_scraping(df_projects):
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Loop through projects with progress updates
    for i, (idx, project) in enumerate(df_projects.iterrows()):
        progress = int(100 * (i + 1) / total_projects)
        progress_bar.progress(progress)
        status_text.text(f"Scraping data for project: {project['project']} ({i+1}/{total_projects})")
        # ... processing ...
```

3. **Fallback Mechanisms**: When primary data sources fail, the system falls back to alternatives:

```python
try:
    # Try to load existing data
    df = pd.read_csv("project_data.csv")
    return df
except:
    # Generate new sample data if file doesn't exist
    # ... data generation logic ...
```

4. **Session State Management**: The application uses Streamlit's session state to maintain context across interactions:

```python
# Initialize session state variables if they don't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_project' not in st.session_state:
    st.session_state.selected_project = None
```

### LLM Integration Workflow

The system integrates multiple large language models through a standardized workflow:

1. **Initialization**: Set up the model with appropriate parameters and API keys
2. **Context Preparation**: Format project data and user queries for the model
3. **Generation**: Query the model with contexts and constraints
4. **Response Processing**: Extract, format, and present insights to the user

For the Gemini integration specifically:

```python
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
```

## How It Works

The application follows this operation flow:

1. **Initialization**:
   - Load or generate project and news data
   - Initialize AI models and session state
   - Set up UI components

2. **Dashboard View**:
   - Display overview of all projects with risk metrics
   - Visualize risk scores and factor breakdowns
   - Allow refresh of data with simulated scraping

3. **Project Details**:
   - Show in-depth analysis of selected project
   - Display risk factors, metrics, and historical trends
   - Provide AI-generated risk summaries

4. **Market Analysis**:
   - Present relevant market news affecting projects
   - Show sentiment analysis and impact on risk profiles
   - Visualize market-related risk adjustments

5. **AI Chatbot**:
   - Take user queries about project risks
   - Provide context-aware responses with risk insights
   - Suggest mitigation strategies based on project data

## Getting Started

### Prerequisites

- Python 3.8+
- Streamlit
- Required packages: pandas, plotly, numpy, google-generativeai, textblob, python-dotenv

### Installation

1. Clone the repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Set up the environment variables:
   - Create a `.env` file with your API keys
   - Or use Streamlit secrets management

### Running the Application

```
streamlit run app.py
```

## Advanced Usage

### Adding Custom Risk Factors

To incorporate additional risk factors, modify the `calculate_risk_score()` function in `risk_tools.py`:

```python
def calculate_risk_score(delay_days, payment_status, resignations, new_factor):
    # Base calculation
    score = (delay_days * 2) + (resignations * 5)
    
    # Payment status impact
    if payment_status == "Late":
        score += 10
    elif payment_status == "Missed":
        score += 20
    
    # Add new factor impact
    score += new_factor * 2
    
    return min(100, score)
```

### Integrating Additional LLMs

The system supports multiple LLMs through its utility functions. To add a new model:

1. Create initialization and generation functions in the utils module
2. Add import statements in the main application
3. Update the UI to include model selection options

## Conclusion

The AI Project Risk Management System provides a comprehensive solution for monitoring and mitigating project risks through data visualization, market analysis, and AI-powered insights. Its modular architecture and robust error handling through callbacks ensure reliable operation even in challenging conditions.
