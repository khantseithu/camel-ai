import os
from dotenv import load_dotenv
from camel.societies import RolePlaying
from camel.types import TaskType, ModelType, ModelPlatformType
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
    api_key=openai_api_key
)

task_kwargs = {
    'task_prompt': 'Develop a plan to TRAVEL TO THE PAST and make changes.',
    'with_task_specify': True,
    'task_specify_agent_kwargs': {'model': model}
}



user_role_kwargs = {
    'user_role_name': 'an ambitious aspiring TIME TRAVELER',
    'user_agent_kwargs': {'model': model}
}

assistant_role_kwargs = {
    'assistant_role_name': 'the best-ever experimental physicist',
    'assistant_agent_kwargs': {'model': model}
}

society = RolePlaying(
    **task_kwargs,
    **user_role_kwargs,
    **assistant_role_kwargs
)



def is_terminated(res):
    if res.terminated:
        print(f"Terminated: {res.terminated}")
        print(f"Terminated Reason: {res.terminated_reason}")
        print(f"Terminated Details: {res.terminated_details}")
        role = res.msg.role_type.name
        reason = res.info['termination_reason']
        print(f"Role: {role}, Reason: {reason}")
    return res.terminated
    
def run(society, round_limit: int=10):
    input_msg = society.init_chat()

    # Starting the interactive session
    for _ in range(round_limit):

        # Get the both responses for this round
        assistant_response, user_response = society.step(input_msg)

        # Check the termination condition
        if is_terminated(assistant_response) or is_terminated(user_response):
            break

        # Get the results
        print(f'[AI User] {user_response.msg.content}.\n')
        # Check if the task is end
        if 'CAMEL_TASK_DONE' in user_response.msg.content:
            break
        print(f'[AI Assistant] {assistant_response.msg.content}.\n')



        # Get the input message for the next round
        input_msg = assistant_response.msg

    return None

run(society)