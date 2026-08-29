import torch

from ghost import Config
from ghost.neurons import RecurrentSNN


def test_persistent_snn_state_persists_and_resets_per_world() -> None:
    torch.manual_seed(5)
    config = Config(worlds=2, hidden_dim=6, snn_ticks=3)
    core = RecurrentSNN(4, 6, config, persistent=True)
    value = torch.ones(2, 4)

    core(value)
    first_membrane = core.mem.clone()
    core(value)
    second_membrane = core.mem.clone()
    assert not torch.equal(first_membrane, second_membrane)

    core.reset(torch.tensor([True, False]))
    assert torch.count_nonzero(core.mem[0]) == 0
    assert torch.count_nonzero(core.spk[0]) == 0
    assert torch.count_nonzero(core.mem[1]) > 0


def test_nonpersistent_snn_does_not_store_state() -> None:
    config = Config(worlds=2, hidden_dim=6, snn_ticks=2)
    core = RecurrentSNN(4, 6, config, persistent=False)
    core(torch.ones(2, 4))
    assert core.mem is None
    assert core.spk is None
