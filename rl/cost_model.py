import torch

class CostModel:
    """Calculates the cost of a forward pass."""
    def __init__(self, num_experts, time_weight=1.0, mem_weight=1.0, flops_weight=1.0):
        self.num_experts = num_experts
        self.time_weight = time_weight
        self.mem_weight = mem_weight
        self.flops_weight = flops_weight

    def calculate_cost(self, measurements, gate_indices):
        """Calculates the cost based on the measurements and the number of active experts."""
        num_active_experts = gate_indices.shape[1]
        
        # Normalize the cost components
        time_cost = measurements['time_ms'] / 1000.0 # Normalize to seconds
        mem_cost = measurements['mem_mb'] / 1024.0 # Normalize to GB
        flops_cost = measurements['flops'] / 1e9 # Normalize to GFLOPs

        # Calculate the weighted cost
        cost = (
            self.time_weight * time_cost +
            self.mem_weight * mem_cost +
            self.flops_weight * flops_cost
        )

        # Add a penalty for the number of active experts
        cost += num_active_experts / self.num_experts

        return cost
