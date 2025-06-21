from Controller import CallModel
from typing import TypedDict,List,Union
from langchain_core.messages import HumanMessage,AIMessage
from langgraph.graph import StateGraph,START,END



class AgentState(TypedDict):
    messages:List[Union[HumanMessage,AIMessage]]

def process(state:AgentState)->AgentState:
    """THis is solve the ai messeage"""
    response = CallModel.callModel({"messages": state['messages']})
    state['messages'].append(response["messages"][0]) 
    print(response["messages"][0].content)
    return state


graph = StateGraph(AgentState)
graph.add_node('process',process)
graph.add_edge(START,'process')
graph.add_edge('process',END)
agent =graph.compile()


conversation_history =[]
user_input =input("Enter promt:")
while user_input !='exit':
    conversation_history.append(HumanMessage(content=user_input))
    result =agent.invoke({'messages':conversation_history})
    # print(result['messages'])
    conversation_history=result['messages']
    user_input =input('Enter:')

with open("log.txt", 'w', encoding='utf-8') as file:
    file.write("Agent Conversational log:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"Human: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of conversation.")

print('log saved')

