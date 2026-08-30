import yaml
import os
import networkx as nx
from inference.base.graph import Graph
from pyvis.network import Network
from agent.agent_info.actions import REASONING_ACTION_LIST, TOOL_ACTION_LIST, TERMINATION_ACTION_LIST

# 使用绝对路径加载配置，避免依赖当前工作目录
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "../.."))
_GLOBAL_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "global.yaml")

class ActionGraph(Graph):
    def __init__(self):
        super().__init__()
        self.REASONING_ACTION_LIST = REASONING_ACTION_LIST
        self.TOOL_ACTION_LIST = TOOL_ACTION_LIST
        self.TERMINATION_ACTION_LIST = TERMINATION_ACTION_LIST
        try:
            with open(_GLOBAL_CONFIG_PATH, "r", encoding="utf-8") as _f:
                global_config = yaml.safe_load(_f)
        except FileNotFoundError:
            global_config = {}
        external_tools_enabled = global_config.get("external_tools_enabled")
        if external_tools_enabled:
            self.actions_collection = REASONING_ACTION_LIST + TOOL_ACTION_LIST + TERMINATION_ACTION_LIST
        else:
            self.actions_collection = REASONING_ACTION_LIST + TERMINATION_ACTION_LIST


    def add_action(self, action_id, action_data, agent_data):
        self._add_node({"id": action_id, "action": action_data, "agent": agent_data})

    def add_dependency(self, from_action_id, to_action_id):
        self._add_edge(from_action_id, to_action_id, len(self._edges))

    def visualize(self, path="action_graph.html"):
        G = nx.DiGraph()
        nodes_colors = []
        for node in self._nodes:
            G.add_node(node["id"], label=node["action"]["action"]["action"] + "\n" + node["agent"], 
                       status=node["action"]["success"], 
                       color="green" if node["action"]["success"] == "Success" else "red")
            nodes_colors.append("green" if node["action"]["success"] == "Success" else "red")
        for edge in self._edges:
            G.add_edge(edge.u, edge.v)
        net = Network(notebook=True, height="750px", width="100%", bgcolor="#FFFFFF", font_color="black", directed=True)
        net.from_nx(G)
        net.show(path)

    def get_action_data(self, action_id):
        for node in self._nodes:
            if node["id"] == action_id:
                return node
        return None
    
    def get_dependencies(self, action_id):
        return [edge.v for edge in self._edges if edge.u == action_id]