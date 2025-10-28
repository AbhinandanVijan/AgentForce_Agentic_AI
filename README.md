# Project Title
CrewAi Salesforce Integration

## Description
This project integrates Salesforce's AI capabilities with CrewAI, allowing for intelligent routing and handling of user queries. 

Salesforce Agent - Build on Coral Cloud Resort Trailhead

## Installation
1. Clone the repository.
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```
4. Install the required packages using:
   ```
   pip install -r requirements.txt
   ```

## Usage
Run the main application to interact with the Salesforce agent:
```
python crew_salesforce_tool_app.py
```

## Components
- **salesforce_llm_adapter.py**: Salesforce Agent Adapter for LLM usage.
- **salesforce_crew_llm.py**: Wrapper for using Salesforce Agent as LLM.
- **salesforce_agent_tool.py**: Wrapper to use Salesforce  Agent as a tool with CrewAi Agent.
- **salesforce_agent_API.py**: Handles API interactions with Salesforce.
- **crew_salesforce_tool_app.py**: Main application for interaction using Salesforce Agent as a tool.
- **crew_salesforce_agent_interaction.py**: Main application for interactions between Crew AI Agent and Salesforce agent as LLM.

## Contributors
- Abhinandan Vijan
