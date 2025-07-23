from langgraph.graph import MessageGraph
from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import HumanMessage, SystemMessage
import json



class Agent():
    workflow = None
    def __init__(self):
        self.workflow = MessageGraph()
        self.workflow.add_node("generator", self.task_generator)
        self.workflow.add_node("worker", self.worker_agent)
        self.workflow.add_node("aggregator", self.aggregator_agent)

        self.workflow.add_edge("generator", "worker")
        self.workflow.add_edge("worker", "aggregator")
        self.workflow.set_entry_point("generator")
        self.workflow.set_finish_point("aggregator")



    # 定义Agents
    def task_generator(self,state):
        return SystemMessage(content=json.dumps({"tasks": [{"id": 1, "data": "A"}, {"id": 2, "data": "B"}]}), role='generator')


    def worker_agent(self,state):
        def process(task):
            return f"Processed {task['data']}"

        with ThreadPoolExecutor() as executor:
            content = json.loads( state[-1].content)
            tasks = content['tasks']
            results = list(executor.map(process, tasks))
        return SystemMessage(content=json.dumps({"results": results}), role='worker')


    def aggregator_agent(self,state):
        content = json.loads(state[-1].content)
        results = content['results']
        summary = "\n".join(results)
        return SystemMessage(content=json.dumps({"final_output": summary}), role='aggregator')

        # 构建工作流
    def run(self, input):
        app = self.workflow.compile()
        result = app.invoke(HumanMessage(role="user",content=input))
        # content = json.loads(result[-1].content)
        # results = content['results']
        return result