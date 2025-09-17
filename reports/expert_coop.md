# Phase 3: Report on Expert Cooperation

This report summarizes the results of training the expert cooperation graph as defined in Phase 3 of the task list.

## 1. Summary

The training was successful after switching from dummy data to the CIFAR-10 dataset. The model achieves a reasonable accuracy (~48%), and the learnable expert graph (adjacency matrix) shows significant development.

The key finding is that with a sufficiently high L1 regularization penalty (`lambda=0.1`), the adjacency matrix becomes extremely sparse. This fulfills the done criteria for the task "2. Training des Kooperations‑Graphen".

## 2. Adjacency Matrix Heatmaps

The following heatmaps visualize the state of the expert adjacency matrix before and after training.

### Initial Matrix (Random)

![Initial Adjacency Matrix](./graphs/adjacency_matrix_epoch_initial.png)

### Final Matrix (After 10 Epochs)

![Final Adjacency Matrix](./graphs/adjacency_matrix_epoch_10.png)

As is clearly visible, the L1 penalty forced the vast majority of weights to zero, creating a sparse cooperation graph.

## 3. Routing Examples

The following is a sample of the expert routing decisions for a batch from the test set after 10 epochs. It shows the top-k indices of the selected experts and the final weights used for aggregation.

```json
{"sample": 0, "indices": [7, 6], "weights": [-0.023845680058002472, -0.5943248271942139, 0.020148981362581253, 0.034253619611263275, -0.24800525605678558, -0.12826967239379883, 0.07681803405284882, 0.13384170830249786]}
{"sample": 1, "indices": [0, 1], "weights": [0.5253820419311523, 0.10127469897270203, -0.6570695638656616, -0.22931239008903503, -0.24660004675388336, -0.2943820655345917, -0.3741764426231384, -0.6429430246353149]}
{"sample": 2, "indices": [0, 7], "weights": [0.3560052216053009, 0.01902751624584198, -0.6170579791069031, -0.28332725167274475, -0.1411488950252533, -0.05112249404191971, -0.2220308482646942, 0.2452421933412552]}
{"sample": 3, "indices": [7, 1], "weights": [0.003640398383140564, 0.149976909160614, -0.3614552915096283, -0.28396525979042053, -0.04866465926170349, -0.4287327527999878, -0.33094361424446106, 0.2714793086051941]}
{"sample": 4, "indices": [6, 0], "weights": [0.39593783020973206, -0.3637765944004059, -0.02530112862586975, 0.18493667244911194, -0.036988161504268646, -0.028645697981119156, 0.49050024151802063, 0.2036648690700531]}
```