from langchain.agents import create_tool_calling_agent
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_chroma import Chroma
from langchain_experimental.utilities import PythonREPL
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langgraph.graph import MessageGraph
from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import json
from config import DASHSCOPE_API_KEY


class Workflow:
    workflow = None
    llm = ChatOpenAI(
    model="qwen-plus",
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    repl = PythonREPL()
    vector_store = Chroma(
        collection_name='demo',
        embedding_function=HuggingFaceEmbeddings(model_name=r'D:\EmbaddingModels\bge-small-zh-v1.5'),
        persist_directory='./chroma_langchain_db'
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 简单合并上下文
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  # 检索前3个相关块
        return_source_documents=True  # 显示参考来源
    )
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'run_python',
                'description': '当你想用print()来获取结果时，请用它',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'input': {
                            'type': 'string',
                            'description': '以print(开头的python语言'
                        }
                    },
                    'required': ['input']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'query_db',
                'description': '查阅华娴信息的时候可以用这个方法',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'input': {
                            'type': 'string',
                            'description': '信息具体内容'
                        }
                    },
                    'required': ['input']
                }
            }
        }
    ]
    # tools_prompt = ChatPromptTemplate.from_messages([
    #     ("system", """你是一个工具选择员，用户传入指令后，你只需要按照用户输入内容根据现有的工具分配任务，
    #                 如果和华娴相关的可以用query_db，
    #                 如果和python相关且带有print内容的用run_python
    #                 如果上面都找不到，则通过llm_search用找结果"""),
    #     ("user", "{input}")
    # ])
    # tools_prompt = """
    #             你是一个工具选择员，用户传入指令后，你只需要按照用户输入内容根据现有的工具分配任务，
    #             如果和华娴相关的可以用query_db，
    #             如果和python相关且带有print内容的用run_python
    #             如果上面都找不到，则通过llm_search用找结果"""
    # tool_agent = create_tool_calling_agent(llm, tools=tools,prompt=tools_prompt)
    # tool_executor = AgentExecutor(agent=tool_agent, tools=tools)
    # tool_agent = create_react_agent(llm, tools=tools)




    def __init__(self):
        self.workflow = MessageGraph()
        self.workflow.add_node("toolSelector", self.tool_selector)
        self.workflow.add_node("responser", self.responser)

        self.workflow.add_edge("toolSelector", "responser")
        self.workflow.set_entry_point("toolSelector")
        self.workflow.set_finish_point("responser")


    def run(self, input):
        app = self.workflow.compile()
        # prompt = """
        #             你是一个工具选择员，用户传入指令后，你只需要按照用户输入内容根据现有的工具分配任务，
        #             如果和华娴相关的可以用query_db，
        #             如果和python相关且带有print内容的用run_python
        #             如果上面都找不到，则通过llm_search用找结果
        #             用户输入：
        #             {input}
        #                                                                             """
        # content = prompt.format(input)
        result = app.invoke(HumanMessage(role="user",content=input))
        return result

    def tool_selector(self, messages):
        completion = self.client.chat.completions.create(
            model="qwen-plus",
            # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=messages,
            tools=self.tools
        )
        choices = completion.model_dump()['choices'][0]['message']
        tool_info = {'content':"没有支持的信息"}
        if choices:
            if choices['tool_calls']:
                if choices['tool_calls'][0]['function']['name'] == 'run_python':
                    tool_info = {"name": "run_python", "role": "tool"}
                    # 提取位置参数信息
                    run_python_input = json.loads(choices['tool_calls'][0]['function']['arguments'])['input']
                    tool_info['content'] = self.run_python(run_python_input)
                if choices['tool_calls'][0]['function']['name'] == 'query_db':
                    tool_info = {"name": "query_db", "role": "tool"}
                    # 提取位置参数信息
                    query_db_input = json.loads(choices['tool_calls'][0]['function']['arguments'])['input']
                    tool_info['content'] = self.query_db(query_db_input)

        return SystemMessage(role='tool', content=tool_info['content'], name=tool_info['name'] if 'name' in tool_info else 'None')


    def responser(self, messages):
        prompt = f"""
        请根据下列信息，找到用户问了什么问题，我们找到了什么答案。把返回格式里XXXX替换成你找到的内容。
        信息：
        {messages}
        返回格式：
        面试官你好，根据你XXXX的问题，我们找到XXXX答案
        """
        completion = self.client.chat.completions.create(
            model="qwen-plus",
            # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[{
                "content": prompt,
                "role": "user"
            }]
        )
        result = completion.model_dump()['choices'][0]['message']['content']
        return SystemMessage(role='system', content=json.dumps({'AI':result}))


    def run_python(self, input):
        return self.repl.run(input)

    def query_db(self, input):
        result = self.qa_chain.invoke({"query": input})
        return result['result']

