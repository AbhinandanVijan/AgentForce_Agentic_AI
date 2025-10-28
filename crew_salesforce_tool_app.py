# crew_app.py
from crewai import Agent, Task, Crew, Process, LLM
from salesforce_agent_tool import SalesforceAgentCrewTool
from dotenv import load_dotenv
import os

load_dotenv()

sf_tool = SalesforceAgentCrewTool()

# ✅ Correct model id + attach HF token
llm = LLM(
    provider="huggingface",
    model= "huggingface/meta-llama/Meta-Llama-3-8B-Instruct",
    api_key=os.getenv("HF_TOKEN"),
    max_tokens=512,
    # optional tuning:
    # temperature=0.7, top_p=0.9
)

# --- Agents (attach llm so any Crew using them inherits it) ---
router = Agent(
    role="Intent Router",
    goal="Decide if the user needs the Wedding flow or the Vacation flow.",
    backstory="You are the front-desk dispatcher. You ONLY decide the flow.",
    llm=llm,           # 👈 important
    verbose=True
)

wedding_planner = Agent(
    role="Wedding Planner",
    goal="Propose 2–3 wedding package options with add-ons.",
    backstory="Destination wedding specialist.",
    tools=[sf_tool],
    llm=llm,           # 👈 important
    verbose=True
)

vacation_planner = Agent(
    role="Vacation Planner",
    goal="Propose a short family vacation plan (rooms + experiences).",
    backstory="Family-centric curator.",
    tools=[sf_tool],
    llm=llm,           # 👈 important
    verbose=True
)

# --- Tasks ---
t_route = Task(
    description=(
        "Read the user's message and return ONLY one word:\n"
        "- 'wedding' if the query is about weddings/packages/ceremonies/receptions, OR\n"
        "- 'vacation' if the query is about short stays/rooms/experiences/family trips."
    ),
    agent=router,
    expected_output="Exactly one word: 'wedding' or 'vacation'. No extra text."
)

t_wedding = Task(
    description=(
        "Check 2–3 wedding package options (Intimate/Classic/Luxury or Custom) with add-ons "
        "(hair & makeup, photography w/ drone, floral, menu, beverages). "
    ),
    agent=wedding_planner,
    expected_output="A concise proposal with 2–3 package options."
)

t_vacation = Task(
    description=(
        "Check a 2–5 day family vacation plan with room options and day-wise experiences "
        "(e.g., historical tour, culinary experience, seaside dinner). "
    ),
    agent=vacation_planner,
    expected_output="A concise plan (rooms + experiences by day)"
)

def _to_text(x) -> str:
    return str(x).strip().lower()

def kickoff(user_message: str):
    # 1) Route (✅ ensure llm is set here too; either via Agent or Crew)
    route_crew = Crew(
        agents=[router],
        tasks=[t_route],
        process=Process.sequential,
        llm=llm,        # redundant since agent has llm, but safe
        verbose=True
    )
    route_raw = route_crew.kickoff(inputs={"user_message": user_message})
    route = _to_text(route_raw)

    if "wedding" in route:
        chosen_task = t_wedding
        chosen_agents = [wedding_planner]
    elif "vacation" in route:
        chosen_task = t_vacation
        chosen_agents = [vacation_planner]
    else:
        guess = "wedding" if any(k in user_message.lower() for k in ["wedding", "bride", "ceremony", "reception"]) else "vacation"
        chosen_task = t_wedding if guess == "wedding" else t_vacation
        chosen_agents = [wedding_planner] if guess == "wedding" else [vacation_planner]

    # 2) Run selected flow (✅ llm attached)
    flow_crew = Crew(
        agents=chosen_agents,
        tasks=[chosen_task],
        process=Process.sequential,
        llm=llm,
        verbose=False
    )
    return flow_crew.kickoff(inputs={"user_message": user_message})

if __name__ == "__main__":
    msg = "Family of 4, 3-night stay next month with a historical tour and a seaside dinner."
    # msg = "We’re planning a March beach wedding for ~50 guests, vegan menu, drone photography."
    print(kickoff(msg))
