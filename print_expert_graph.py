
import torch
from models.expert_graph import ExpertGraph

# This script is for visualizing the adjacency matrix as per the task list.

if __name__ == "__main__":
    NUM_EXPERTS = 8
    FEATURE_DIM = 16 # Example dimension

    # Instantiate the graph
    expert_graph = ExpertGraph(num_experts=NUM_EXPERTS, feature_dim=FEATURE_DIM)

    print("--- Initial Expert Adjacency Matrix ---")
    expert_graph.print_graph()

    # In a real scenario, you would load a trained model's state_dict.
    # For example:
    # model.load_state_dict(torch.load('checkpoints/expert_graph_best.ckpt'))
    # expert_graph.print_graph()
