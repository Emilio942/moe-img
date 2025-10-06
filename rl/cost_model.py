import torch

class CostModel:
    """Calculates the cost of a forward pass.

    input_mode:
      - 'raw': expects measurements in raw units {time_ms, mem_mb, flops}
      - 'normalized': expects measurements already normalized (z/minmax); uses as-is
    """
    def __init__(self, num_experts, time_weight=1.0, mem_weight=1.0, flops_weight=1.0, input_mode: str = 'raw'):
        self.num_experts = num_experts
        self.time_weight = time_weight
        self.mem_weight = mem_weight
        self.flops_weight = flops_weight
        if input_mode not in ('raw', 'normalized'):
            raise ValueError("input_mode must be 'raw' or 'normalized'")
        self.input_mode = input_mode

    def calculate_cost(self, measurements, gate_indices):
        """Calculates the cost based on measurements and number of active experts.

        measurements: dict with keys 'time_ms','mem_mb','flops' (raw) or normalized equivalents
        gate_indices: tensor shape (B, k) used for active expert count penalty
        """
        num_active_experts = gate_indices.shape[1]

        if self.input_mode == 'raw':
            # Convert to comparable scales
            time_cost = measurements.get('time_ms', 0.0) / 1000.0  # seconds
            mem_cost = measurements.get('mem_mb', 0.0) / 1024.0   # GB
            flops_cost = measurements.get('flops', 0.0) / 1e9     # GFLOPs
        else:  # 'normalized'
            time_cost = measurements.get('time_ms', 0.0)
            mem_cost = measurements.get('mem_mb', 0.0)
            flops_cost = measurements.get('flops', 0.0)

        # Weighted sum
        cost = (
            self.time_weight * float(time_cost) +
            self.mem_weight * float(mem_cost) +
            self.flops_weight * float(flops_cost)
        )

        # Penalty for number of active experts (per-batch average factor)
        cost += num_active_experts / max(1, self.num_experts)

        return float(cost)
