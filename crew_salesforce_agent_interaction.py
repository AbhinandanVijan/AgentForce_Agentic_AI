# crew_salesforce_app.py
import os, re
from dotenv import load_dotenv, find_dotenv
from crewai import Agent, Task, Crew, Process, LLM


from salesforce_agent_API import SalesforceAgentAPI
from salesforce_crew_llm import SalesforceCrewLLM

load_dotenv(find_dotenv(), override=False)

# ---------- LLMs ----------
# Customer "mind" (any provider you prefer; keeping HF 8B w/ tight tokens)
customer_llm = LLM(
    provider="huggingface",
    model="huggingface/meta-llama/Meta-Llama-3-8B-Instruct",
    api_key=os.getenv("HF_TOKEN"),
    max_tokens=220,
    temperature=0.7,
)

# Coral Cloud agent "mind" (Salesforce Einstein via your bridge)
sf_llm = SalesforceCrewLLM(api=SalesforceAgentAPI())

# ---------- Agents ----------
# 1) Customer Simulator: generates one crisp, on-topic resort question
customer = Agent(
    role="Guest Trip Planner",
    goal=(
        "Compose a single, clear question to Coral Cloud Resort to plan a stay, "
        "including relevant details (party size, dates or timeframe, room type, "
        "amenities/activities, dining preferences) that are likely covered by resort FAQs."
    ),
    backstory=(
        "You are a prospective guest planning a vacation at Coral Cloud Resort. "
        "Stay strictly within resort-relevant topics: rooms/suites, check-in/out, amenities "
        "(pool, spa, fitness, kids club), water sports/activities, resort/local tours offered "
        "through the resort, on-site dining (including seaside dinners), basic transport/parking, "
        "policies (cancellations, deposits), and how to start booking. "
        "Avoid non-resort topics such as flights, third-party attractions, car rentals, or discounts "
        "not mentioned in FAQs."
    ),
    llm=customer_llm,
    allow_delegation=False,
    verbose=False,
    memory=False,
)

# 2) Coral Cloud FAQ Agent: answers only if within FAQ scope
coral_cloud = Agent(
    role="Coral Cloud FAQ Specialist",
    goal=(
        "Answer Coral Cloud Resort FAQ questions accurately, concisely, and helpfully. "
        "Stay strictly within published FAQs and allowed service information."
    ),
    backstory=(
        "You are the embedded Coral Cloud Resort assistant connected to Salesforce Einstein Agent. "
        "You ONLY cover official FAQ topics (rooms, rates/policies, check-in/out, amenities, dining, "
        "spa & fitness, water sports/activities, family options, resort/local tours offered by the resort, "
        "parking/transport basics, contact/booking steps). "
        "If a request is outside FAQs or requires staff action (custom packages, price quotes not listed, "
        "availability holds), say you can only handle FAQs and provide the correct next steps (phone/email/booking link). "
        "Keep answers friendly and structured with short headings and bullet points."
    ),
    llm=sf_llm,
    allow_delegation=False,
    verbose=False,
    memory=False,
)

# ---------- Tasks ----------
customer_task = Task(
    description=(
        "Create ONE guest message to Coral Cloud Resort asking about planning a vacation. "
        "Must include:\n"
        "- party size (e.g., 2 adults + 2 kids),\n"
        "- stay length/timeframe (e.g., 3 nights in late November),\n"
        "- at least two interest areas (e.g., family suite, kids club, water sports, historical tour, seaside dinner).\n\n"
        "Constraints:\n"
        "- Only resort-relevant topics (rooms, amenities, dining, spa, activities, tours offered by the resort, check-in/out, policies, basic transport/parking, how to book).\n"
        "- 1–2 sentences max. No preamble, no bullet points. Return ONLY the message text you would send to the resort."
    ),
    agent=customer,
    expected_output="A single-line or two-sentence guest question suitable to send to Coral Cloud Resort.",
)

coral_cloud_task = Task(
    description=(
        "You are Coral Cloud’s FAQ assistant. Reply to the guest message below.\n\n"
        "Rules:\n"
        "- If info is missing, ask up to 2 brief clarifying questions first, then answer.\n"
        "- If the request is outside FAQs or needs staff intervention, say so and provide contact/next steps.\n"
        "- Use short headings and bullet points; keep it friendly and concise.\n\n"
        "Guest message:\n{customer_message}"
    ),
    agent=coral_cloud,
    expected_output=(
        "A friendly, structured answer with short headings and bullet points. "
        "If out of scope, a brief note plus contact/booking steps."
    ),
)

# ---------- Crew orchestration ----------
crew = Crew(
    agents=[customer, coral_cloud],
    tasks=[customer_task, coral_cloud_task],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    # Run the first task to get the customer's message
    result1 = crew.kickoff()  # runs customer_task then coral_cloud_task with no inputs yet    

    # Always close Salesforce session at the end
    try:
        sf_llm.close()
    except Exception:
        pass