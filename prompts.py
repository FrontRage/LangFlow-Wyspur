executive_summary_prompt = """\
📄 Executive Summary: Create a detailed executive summary that covers:
🎯 Key Outcomes: Summarize the primary objectives achieved and major insights gained.
🔄 Strategic Decisions: Document significant decisions made and their expected impact on future directions.
Be as brief as possible.\
Meeting Transcript:
[MEETING TRANSCRIPT HERE{transcript}]

Onboarding document:
[MEETING TRANSCRIPT HERE{onboarding}]
"""

meeting_minutes_prompt = """\
📄 Meeting Minutes: Develop comprehensive meeting minutes including:
📋 Attendees: List all participants along with their roles and departments.
🗣️ Discussion Points: Detail the topics discussed, including any debates or alternate viewpoints.
✅ Decisions Made: Record all decisions, including who made them and the rationale.
📌 Action Items: Specify tasks assigned, responsible individuals, and deadlines.
📊 Data & Insights: Summarize any data presented or insights shared that influenced the meeting's course.
🔄 Follow-Up: Note any agreed-upon follow-up meetings or checkpoints.
Meeting Transcript:
[MEETING TRANSCRIPT HERE{transcript}]

Onboarding document:
[MEETING TRANSCRIPT HERE{onboarding}]\
"""

sales_discovery_prompt = """\
📄 Sales Discovery Summary: Compile a thorough summary that includes:
🎯 Client Needs: Describe the client's expressed needs, challenges, and goals.
🛠️ Proposed Solutions: Detail potential solutions discussed to address the client's needs.
🤔 Questions & Clarifications: List any questions the client had and the clarifications provided.
✅ Action Items: Outline next steps for both the sales team and the client, including any documents or information to be exchanged.
💬 Feedback: Note initial client feedback or reactions to the proposed solutions.
Meeting Transcript:
[MEETING TRANSCRIPT HERE{transcript}]

Onboarding document:
[MEETING TRANSCRIPT HERE{onboarding}]
"""

sales_meeting_prompt = """\
📄 Sales Meeting Overview: Craft a detailed overview that encompasses:
🎯 Meeting Objectives: Clarify the goals set for the sales meeting and whether they were achieved.
📊 Customer Insights: Document insights into customer behavior, preferences, or feedback mentioned during the meeting.
🛠️ Product Updates: Summarize any new product features, updates, or offerings discussed.
✅ Action Items: List detailed action items for the team, specifying roles, responsibilities, and timelines.
💡 Decisions: Highlight key decisions made regarding strategies, pricing, or customer approach.
📈 Performance Metrics: Review any sales performance metrics or targets reviewed during the meeting.
Meeting Transcript:
[MEETING TRANSCRIPT HERE{transcript}]

Onboarding document:
[MEETING TRANSCRIPT HERE{onboarding}]
"""

customer_success_prompt = """\
📄 Customer Success Meeting Insights: Construct a detailed report that includes:
🎯 Customer Feedback: Capture specific feedback from customers on products or services.
🛠️ Solution Optimization: Discuss solutions or adjustments proposed to enhance customer satisfaction.
📈 Success Metrics: Review key metrics or KPIs that measure customer success and any changes over time.
✅ Action Items: Enumerate actions to be taken to address customer concerns, including responsible parties and deadlines.
🔄 Follow-Up Plans: Outline plans for ongoing customer engagement, including scheduled check-ins or updates.
💡 Innovative Ideas: Note any new ideas or suggestions for improving customer experience or engagement.
Meeting Transcript:
[MEETING TRANSCRIPT HERE{transcript}]

Onboarding document:
[MEETING TRANSCRIPT HERE{onboarding}]
"""

user_research_prompt = """\
📄 User Research Findings: Compile a comprehensive summary that covers:
🎯 Research Objectives: State the goals of the user research and whether they were met.
🧐 Key Insights: Highlight significant findings about user behavior, preferences, and pain points.
📊 Data Analysis: Summarize data collected during the research, including user surveys, interviews, or usability tests.
✅ Action Items: List recommendations for product or service improvements based on research findings.
🔄 Next Steps: Define what further research or validation is needed.
💬 Stakeholder Feedback: Include any feedback from stakeholders on the research findings and proposed actions.
Meeting Transcript:
[MEETING TRANSCRIPT HERE{transcript}]

Onboarding document:
[MEETING TRANSCRIPT HERE{onboarding}]
"""

team_update_prompt = """\
📄 Team Update Summary: Create an in-depth summary that includes:
🎯 Team Achievements: Celebrate successes and milestones achieved by the team.
📈 Progress Tracking: Update on ongoing projects, including current status and any hurdles encountered.
✅ Action Items: Detail specific tasks assigned, including deadlines and responsible individuals.
🔄 Upcoming Goals: Outline goals and objectives for the upcoming period.
💡 Ideas and Suggestions: Encourage sharing of new ideas or suggestions for improving team performance or processes.
🤝 Team Feedback: Summarize feedback or concerns raised by team members and any resolutions proposed.
Meeting Transcript:
[MEETING TRANSCRIPT HERE{transcript}]

Onboarding document:
[MEETING TRANSCRIPT HERE{onboarding}]
"""

training_prompt = """\
📄 Training Session Recap: Provide a thorough recap that encompasses:
🎯 Training Objectives: Outline the goals of the training session and their achievement status.
📚 Content Covered: Detail the key concepts, skills, or knowledge areas covered during the training.
✅ Action Items: Assign follow-up tasks or practice activities to reinforce learning.
📝 Assessments: Summarize any assessments or evaluations conducted and the outcomes.
🔄 Feedback and Improvements: Gather participant feedback on the training session and suggestions for future sessions.
📈 Next Steps: Identify areas for further development or upcoming training needs.
Meeting Transcript:
[MEETING TRANSCRIPT HERE{transcript}]

Onboarding document:
[MEETING TRANSCRIPT HERE{onboarding}]
"""


meeeting_actionable_insights = """You are an expert in extracting actionable insights from meetings. I will provide a meeting transcript and an onboarding questionnaire. Your task is to carefully analyze the conversation and document, and then produce a comprehensive set of outcomes and next steps.  Specifically, please only concentrate on tasks from our end around outreach, messaging, research, or event participation, being the B2B lead generation company offering the services, it should include:

Identify the Core Objectives:
Summarize the main goals, targets, or intentions discussed in the meeting.
Develop Any Relevant Strategies:
Based on the meeting’s context, propose strategies or approaches that can help achieve the identified objectives. If there was discussion about outreach, messaging, research, or event participation, outline how to execute these strategies effectively.
Identify Potential Contacts, Resources, or References:
Suggest any types of contacts to research or connect with (e.g., potential clients, partners, experts, common alumni, previous work contacts, etc), as well as resources (e.g., databases, websites, organizations) that could assist in completing the tasks.
Propose Communication or Outreach Templates (If Applicable):
a.Provide a draft template or messaging framework if the meeting indicates a need for outreach (such as emailing potential partners or drafting proposals). Provide 10 outreach or email variations depending on the strategies outlined in the call. Explain your reasoning for each variation.

b.Additionally provide the relevant filters and keywords to create the target audiences in Linkedin Sales Navigator for each of the target audiences discussed in the call.


Events and Opportunities (If Applicable):
If the meeting mentions or implies events, conferences, or workshops, list any potentially relevant ones, along with details such as location, timing, and why they are important. Provide strategy on how to tackle this even based on the product fit and target audience.
Important: Do not include the transcript in your final answer. Instead, use the transcript content to inform your response. If certain details are not provided, make reasonable assumptions based on standard business practices or typical approaches in similar scenarios.
Meeting Transcript:
[MEETING TRANSCRIPT HERE{transcript}]

Onboarding document:
[MEETING TRANSCRIPT HERE{onboarding}]
"""

research_agent_queries = """
You are an assistant that reads a transcript of a call. In this call, certain research tasks or instructions for information gathering may have been mentioned.

Your job:
1. Identify all instructions that indicate the need for further research or external information. Consider relevant topics for you to reaserch based on the following filter criteria:
    - Needs to be something that can be researched online by an LLM, this queries will be sent to a research agent.
    - The research has to be relevant to delivering the service provided in the call, meaning could help the B2B lead generation company understand the industry or product of the client better and will help aid targeting.
    - Anything that client mentions as something that will add value to their business that the B2B lead gen company could do in its behalf to add value to the conversation.
    - Dont include doing research on tools or topics that are related to setup of platforms, etc
2. For each such instruction, generate a search query that can be used in a web search.
3. Make sure the queries capture the essence and context of the need in the call, make them industry specific and relevant to delivering the provided value to the client in the call.
4. Return your result as JSON with the following format:

{{
  "queries": [
    {{
      "description": "<a brief natural language description of the research instruction>",
      "query": "<the concise search query>"
    }},
    ...
  ]
}}

Instructions:
- The 'description' should be a short sentence explaining what is being researched, in plain English.
- The 'query' should be a succinct and effective search query that could be plugged into a search engine.
- Include all the relevant queries derived from the transcript. If no actionable research tasks are found, return an empty queries array.

Only return the JSON and nothing else.
[MEETING TRANSCRIPT HERE:

{transcript}]
""".strip()