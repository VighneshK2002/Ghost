# Neuromorphic roadmap

Containing spiking neurons is not the same as running efficiently on
neuromorphic hardware. The current implementation is a PyTorch research model
with several event-compatible ideas and several dense, global operations.

| Component | Current implementation | Suitability | Work required |
|---|---|---:|---|
| Recurrent cores | Multi-tick recurrent LIF, binary spikes, explicit membrane state | Medium–high conceptually | Map supported neuron/synapse dynamics; quantify firing sparsity; remove dense membrane feature dependence; respect fan-in and precision constraints. |
| Surrogate derivative | Triangular PyTorch autograd surrogate | Low for inference; medium as training abstraction | Replace autograd training with a device-supported/local rule or host-device learning protocol. |
| Eligibility traces | Explicit per-world dense tensors, exact/autograd Jacobian assists, TD modulation | Medium conceptually, low as implemented | Factor/localize traces, bound memory, remove batched Jacobian/autograd calls, map modulatory signals to hardware primitives. |
| Encoder/predictor/actor projections | Dense `nn.Linear` layers | Medium | Sparsify/quantize, map to synaptic cores, validate fan-in/fan-out and latency. |
| LayerNorm | Dense PyTorch normalization across features | Low | Replace with fixed scaling, homeostatic mechanisms, or hardware-supported normalization; retrain and revalidate. |
| GELU/Tanh MLPs | Conventional dense nonlinear networks | Low | Replace with spiking/subthreshold circuits or supported approximations. |
| Strategy gate and softmax policy | Sigmoid/tanh gate plus categorical softmax | Low–medium | Implement compatible gating/action selection without dense floating-point softmax. |
| Adam | Host-side PyTorch Adam for several parameter groups | Low | Develop local/on-chip updates or an explicit hybrid training architecture; account for optimizer-state memory. |
| EMA target encoder/critic | Dense duplicated networks with parameter averaging (optional paths) | Low | Avoid duplicate parameter storage or define an efficient host-managed target mechanism. |
| Reward/TD broadcast | Scalar/global modulatory signal | Medium–high conceptually | Define routing, delay, precision, and locality on a target platform. |
| Batched environment | NumPy vectorized host simulation | Not applicable | Build a sensor/action interface and define real-time synchronization with the device. |

## Milestones

1. Profile spike rates, dense operation counts, trace memory, and update traffic
   on the canonical task.
2. Define a target hardware model and its supported neuron, plasticity,
   precision, routing, and reset semantics.
3. Replace LayerNorm/GELU/softmax dependencies and validate behavior after each
   substitution.
4. Reduce eligibility storage to hardware-feasible local state and remove
   episode-spanning autograd/Jacobian dependencies.
5. Quantize weights/state, impose connectivity limits, and retrain with those
   constraints.
6. Compare correctness, latency, energy, and learning behavior on a simulator
   and then physical hardware.

Until those steps are complete, the accurate claim is: Ghost is a PyTorch
architecture using recurrent spiking cores and online eligibility mechanisms,
developed toward neuromorphic execution.
