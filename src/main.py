from dotenv import load_dotenv
import os
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig
from camel.agents import ChatAgent
from camel.toolkits import MathToolkit, SearchToolkit
from camel.messages import BaseMessage
from camel.types import RoleType

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


sys_msg = 'You are a curious stone wondering about the universe.'

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig().as_dict(),
    api_key=OPENAI_API_KEY,
)


agent = ChatAgent(
    system_message=sys_msg,
    model=model,
    message_window_size=10,
    tools=[
        *MathToolkit().get_tools(),
        *SearchToolkit().get_tools(),
    ],
    output_language="korean"
)



user_input = 'What is life?'

new_msg = BaseMessage(
    role_name="camel user",
    content="This is a new user message would add to agent memory",
    role_type=RoleType.USER,
    meta_dict={}
)

agent.record_message(new_msg)


res = agent.step(user_input)
print(res.info['tool_calls']);

print(res.msgs[0].content);

print("memory: ", agent.memory.get_context())

