import numpy as np
import torch

from dex.model import DEX, Config, training_loss
from dex.refine import ResidualDecision


def small():
    torch.manual_seed(29)
    return DEX(Config(vocab=80, width=24, depth=1, heads=3, ff=48, window=16, rank=8)).eval()


def test_parallel_fields_and_option_permutation():
    model = small()
    state, qs, opts = torch.randint(1, 80, (3, 12)), torch.randint(1, 80, (4, 8)), torch.randint(1, 80, (9, 10))
    with torch.no_grad():
        together = model(state, qs, opts)
        singles = torch.cat([model(state, q[None], opts) for q in qs])
        permutation = torch.tensor([8, 1, 3, 4, 0, 6, 2, 5, 7])
        permuted = model(state, qs, opts[permutation])
    torch.testing.assert_close(together, singles, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(permuted, together[:, permutation], atol=1e-6, rtol=1e-5)


def test_neural_lazy_path_agrees_with_dense_forward():
    model = small()
    state, qs, opts = torch.randint(1, 80, (5, 12)), torch.randint(1, 80, (2, 8)), torch.randint(1, 80, (17, 10))
    with torch.no_grad():
        tokens, summary = model.encoder(state, torch.zeros(5, dtype=torch.long))
        _, questions = model.encoder(qs, torch.ones(2, dtype=torch.long))
        _, options = model.encoder(opts, torch.full((17,), 2, dtype=torch.long))
        bank = torch.nn.functional.normalize(options, dim=-1)
        coarse, weights = model.core.prepare(questions, summary)
        full = model.core(questions, summary, tokens, state != 0, bank).softmax(-1).numpy()
        for f in range(2):
            def state_refine(ids):
                ids = torch.from_numpy(ids)
                return model.core.refine_state(questions[f:f+1].expand(len(ids), -1),
                    coarse[f:f+1].expand(len(ids), -1), tokens[ids], (state != 0)[ids]).numpy()
            def option_refine(ids):
                ids = torch.from_numpy(ids)
                return model.core.refine_options(questions[f:f+1].expand(len(ids), -1),
                    coarse[f:f+1].expand(len(ids), -1), bank[ids]).numpy()
            w = weights[f].double().numpy()
            w /= w.sum()  # float64 reference re-normalization, <1e-7 difference.
            runner = ResidualDecision(coarse[f].numpy(), w, bank.numpy(), state_refine, option_refine)
            out = runner.run(tolerance=0)
            np.testing.assert_allclose(out.probabilities, full[f], atol=2e-7)


def test_training_gradient_reaches_all_three_components():
    model = small().train()
    out = model(torch.randint(1, 80, (3, 12)), torch.randint(1, 80, (4, 8)), torch.randint(1, 80, (5, 8)))
    training_loss(out, torch.tensor([0, 1, 2, 3])).backward()
    for weight in (model.encoder.embedding.weight, model.core.route.weight,
                   model.core.state_delta[-1].weight, model.core.option_delta[-1].weight):
        assert weight.grad is not None and torch.isfinite(weight.grad).all()
        assert weight.grad.abs().sum() > 0

