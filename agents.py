from langchain.agents import AgentType,initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import ShellTool

# from langchain_community.llms.mlx_pipeline import MLXPipeline

# llm = MLXPipeline.from_model_id(
#     "YashJain/GitAI-Qwen2-0.5B-Instruct-MLX-v1"
# )

llm=ChatOpenAI(temperature=0,model="gpt-3.5-turbo")

shell_tool=ShellTool()
agent=initialize_agent([shell_tool],llm,agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
agent.run("execture a git commit with message 'somebody help me with mlx' and then push it")