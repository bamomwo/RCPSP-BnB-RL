"""Regression tests for the independent actor/critic PPO architecture."""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts import train_ppo_gpu as ppo
from rcpsp_bb_rl.ml.models import (
    BranchingTransformer,
    SeparateActorCritic,
    load_policy_checkpoint,
    load_separate_actor_critic_checkpoint,
    save_policy_checkpoint,
    save_separate_actor_critic_checkpoint,
)


class SeparateActorCriticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def setUp(self) -> None:
        torch.manual_seed(11)
        self.shared = BranchingTransformer(
            global_dim=3,
            candidate_dim=4,
            d_model=8,
            n_heads=2,
            n_layers=1,
            ffn_dim=16,
            dropout=0.0,
            critic_feature_dim=2,
        )
        self.model = SeparateActorCritic.from_shared(self.shared).train()
        self.candidates = torch.randn(3, 4)
        self.global_feats = torch.randn(3)
        self.critic_feats = torch.randn(2)
        self.mask = torch.ones(3, dtype=torch.bool)

    def test_split_preserves_initial_actor_and_value_outputs(self) -> None:
        self.shared.eval()
        self.model.eval()
        old_logits, old_value = self.shared(
            self.candidates, self.global_feats, self.mask, self.critic_feats,
        )
        new_logits, new_value = self.model(
            self.candidates, self.global_feats, self.mask, self.critic_feats,
        )
        torch.testing.assert_close(new_logits, old_logits)
        torch.testing.assert_close(new_value, old_value)

    def test_value_loss_has_no_actor_gradient(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        before = {k: v.clone() for k, v in self.model.actor.state_dict().items()}
        _, value = self.model(
            self.candidates, self.global_feats, self.mask, self.critic_feats,
        )
        value.backward()
        actor_grads = [p.grad for p in self.model.actor.parameters() if p.requires_grad]
        critic_grads = [p.grad for p in self.model.critic.parameters() if p.requires_grad]
        self.assertTrue(all(g is None or torch.count_nonzero(g) == 0 for g in actor_grads))
        self.assertTrue(any(g is not None and torch.count_nonzero(g) > 0 for g in critic_grads))
        optimizer.step()
        for key, parameter in self.model.actor.state_dict().items():
            torch.testing.assert_close(parameter, before[key], rtol=0, atol=0)

    def test_policy_loss_updates_actor_trunk_without_critic_gradient(self) -> None:
        logits, _ = self.model(
            self.candidates, self.global_feats, self.mask, self.critic_feats,
        )
        loss = -torch.log_softmax(logits, dim=0)[0]
        loss.backward()
        self.assertGreater(self.model.actor.layers[0].attn.in_proj_weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in self.model.critic.parameters()))

    def test_splitting_preserves_rng_stream_and_uses_independent_storage(self) -> None:
        rng_before = torch.get_rng_state()
        model = SeparateActorCritic.from_shared(self.shared)
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
        self.assertTrue(
            {p.data_ptr() for p in model.actor.parameters()}.isdisjoint(
                p.data_ptr() for p in model.critic.parameters()
            )
        )

    def test_large_critic_gradients_do_not_shrink_actor_clipping(self) -> None:
        ac = ppo.ActorCritic(self.model)
        actor = next(self.model.actor.parameters())
        critic = next(self.model.critic.parameters())
        actor.grad = torch.ones_like(actor)
        critic.grad = torch.ones_like(critic)
        ac.clip_gradients(0.5)
        expected = actor.grad.clone()
        actor.grad = torch.ones_like(actor)
        critic.grad = torch.ones_like(critic) * 1e6
        ac.clip_gradients(0.5)
        torch.testing.assert_close(actor.grad, expected, rtol=0, atol=0)

    def test_bc_initialization_matches_shared_path_for_both_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for resource_set in (False, True):
                with self.subTest(resource_set=resource_set):
                    bc = BranchingTransformer(
                        global_dim=3, candidate_dim=4, d_model=8, n_heads=2,
                        n_layers=1, ffn_dim=16, dropout=0.0,
                        resource_feature_dim=2 if resource_set else 0,
                        interaction_feature_dim=3 if resource_set else 0,
                    ).eval()
                    path = Path(tmp) / "bc.pt"
                    save_policy_checkpoint(bc, str(path))
                    with contextlib.redirect_stdout(io.StringIO()):
                        torch.manual_seed(42)
                        shared = load_policy_checkpoint(str(path), critic_feature_dim=2)
                        rng_after_shared = torch.get_rng_state()
                        torch.manual_seed(42)
                        separate = load_separate_actor_critic_checkpoint(
                            str(path), critic_feature_dim=2,
                        )
                    torch.testing.assert_close(torch.get_rng_state(), rng_after_shared)
                    resources = {} if not resource_set else {
                        "resource_feats": torch.randn(2, 2),
                        "candidate_resource_feats": torch.randn(3, 2, 3),
                        "resource_mask": torch.tensor([True, False]),
                    }
                    expected = shared(
                        self.candidates, self.global_feats, self.mask, self.critic_feats,
                        **resources,
                    )
                    actual = separate(
                        self.candidates, self.global_feats, self.mask, self.critic_feats,
                        **resources,
                    )
                    torch.testing.assert_close(actual[0], expected[0])
                    torch.testing.assert_close(actual[1], expected[1])
                    bc_logits, _ = bc(self.candidates, self.global_feats, self.mask, **resources)
                    torch.testing.assert_close(actual[0], bc_logits)

    def test_value_renormalization_preserves_raw_predictions_and_actor(self) -> None:
        ac = ppo.ActorCritic(self.model)
        optimizer = torch.optim.AdamW(ac.parameters(), lr=0.01)
        logits_before, value_before = ac(
            self.candidates, self.global_feats, self.mask, self.critic_feats,
        )
        ppo.renormalize_value_head(
            ac.critic_model, optimizer,
            old_mean=2.0, old_std=3.0, new_mean=5.0, new_std=7.0,
        )
        logits_after, value_after = ac(
            self.candidates, self.global_feats, self.mask, self.critic_feats,
        )
        torch.testing.assert_close(logits_after, logits_before, rtol=0, atol=0)
        torch.testing.assert_close(value_after * 7.0 + 5.0, value_before * 3.0 + 2.0)

    def test_checkpoint_restores_actor_and_critic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "separate.pt"
            save_separate_actor_critic_checkpoint(
                self.model,
                str(path),
                extra={"value_norm": {"mean": 1.0, "var": 2.0, "count": 3.0}},
            )
            restored, value_norm = load_separate_actor_critic_checkpoint(
                str(path), return_value_norm=True,
            )
            expected = self.model(
                self.candidates, self.global_feats, self.mask, self.critic_feats,
            )
            actual = restored(
                self.candidates, self.global_feats, self.mask, self.critic_feats,
            )
            torch.testing.assert_close(actual[0], expected[0])
            torch.testing.assert_close(actual[1], expected[1])
            self.assertEqual(value_norm["count"], 3.0)

            deployment_actor = load_policy_checkpoint(str(path))
            deployment_logits, _ = deployment_actor(
                self.candidates, self.global_feats, self.mask,
            )
            torch.testing.assert_close(deployment_logits, expected[0])
            compiled = torch.jit.freeze(torch.jit.script(deployment_actor.eval()))
            compiled_logits, _ = compiled(self.candidates, self.global_feats, self.mask)
            torch.testing.assert_close(compiled_logits, expected[0])


if __name__ == "__main__":
    unittest.main()
