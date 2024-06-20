from langchain.agents import AgentType,initialize_agent,load_tools
from langchain.chat_models import ChatOpenAI
from langchain.tools import ShellTool
llm=ChatOpenAI(temperature=0,model="gpt-3.5-turbo")
shell_tool=ShellTool()
agent=initialize_agent([shell_tool],llm,agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
agent.run("execute the command for making a git commit with message 'git agent' and then push it")