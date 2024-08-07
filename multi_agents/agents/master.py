import os
import time
from langgraph.graph import StateGraph, END
from .utils.views import print_agent_output
from ..memory.research import ResearchState
from .utils.utils import sanitize_filename
from PIL import Image
from graphviz import Digraph

# 주어진 Graph 객체를 사용하여 nodes와 edges를 추출하는 함수
def extract_nodes_edges(graph):
    # 노드와 엣지를 저장할 리스트와 딕셔너리
    nodes = {}
    edges = []

    # 노드 추출
    for node_id, node in graph.nodes.items():
        nodes[node_id] = node.name

    # 엣지 추출
    for edge in graph.edges:
        edges.append((edge.source, edge.target))

    return nodes, edges


def create_graph_png(nodes, edges, output_file='graph_output.png'):
    """
    Create a PNG image of a graph based on provided nodes and edges.

    Parameters:
    - nodes (dict): A dictionary of nodes where keys are node IDs and values are node names.
    - edges (list): A list of tuples representing edges, where each tuple contains (source, target).
    - output_file (str): The filename for the output PNG image. Defaults to 'graph_output.png'.
    """
    # Create a new Digraph object
    dot = Digraph()

    # Add nodes to the graph
    for node_id, node_name in nodes.items():
        dot.node(node_id, node_name)

    # Add edges to the graph
    for source, target in edges:
        dot.edge(source, target)

    # Render the graph to a PNG file
    dot.render(filename=output_file, format='png', cleanup=True)
    print(f"Graph saved as {output_file}")






# Import agent classes
from . import \
    WriterAgent, \
    EditorAgent, \
    PublisherAgent, \
    ResearchAgent

class ChiefEditorAgent:
    def __init__(self, task: dict, websocket=None, stream_output=None, tone=None, headers=None):
        self.task_id = int(time.time()) # Currently time based, but can be any unique identifier
        self.output_dir = "./outputs/" + sanitize_filename(f"run_{self.task_id}_{task.get('query')[0:40]}")
        self.task = task
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}
        self.tone = tone
        os.makedirs(self.output_dir, exist_ok=True)

    def init_research_team(self):
        # Initialize agents
        writer_agent = WriterAgent(self.websocket, self.stream_output, self.headers)
        editor_agent = EditorAgent(self.websocket, self.stream_output, self.headers)
        research_agent = ResearchAgent(self.websocket, self.stream_output, self.tone, self.headers)
        publisher_agent = PublisherAgent(self.output_dir, self.websocket, self.stream_output, self.headers)

        # Define a Langchain StateGraph with the ResearchState
        workflow = StateGraph(ResearchState)

        # Add nodes for each agent
        workflow.add_node("browser", research_agent.run_initial_research)
        workflow.add_node("planner", editor_agent.plan_research)
        workflow.add_node("researcher", editor_agent.run_parallel_research)
        workflow.add_node("writer", writer_agent.run)
        workflow.add_node("publisher", publisher_agent.run)

        workflow.add_edge('browser', 'planner')
        workflow.add_edge('planner', 'researcher')
        workflow.add_edge('researcher', 'writer')
        workflow.add_edge('writer', 'publisher')

        # set up start and end nodes
        workflow.set_entry_point("browser")
        workflow.add_edge('publisher', END)

        return workflow

    async def run_research_task(self):
        research_team = self.init_research_team()

        # compile the graph
        chain = research_team.compile()

        # Create the graph PNG
        nodes, edges = extract_nodes_edges(chain.get_graph())
        create_graph_png(nodes, edges, output_file='graph_output.png')

        # 이미지 열기
        # img = Image.open('graph_output.png')
        # img.show()

        if self.websocket and self.stream_output:
            print("self.task.get('multi_agents'): ",self.task.get('multi_agents'))
            if self.task.get('multi_agents'):
                # multi agent 
                print_agent_output(f"Starting the research process for query '{self.task.get('query')}'...", "MASTER")
            else:
                # single researcher
                await self.stream_output("logs", "starting_research", f"Starting the research process for query '{self.task.get('query')}'...", self.websocket)
        # else:
            # multi agent 
            # print_agent_output(f"Starting the research process for query '{self.task.get('query')}'...", "MASTER")
 
        result = await chain.ainvoke({"task": self.task})

        return result