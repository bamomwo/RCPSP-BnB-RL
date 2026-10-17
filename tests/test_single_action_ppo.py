"""Regression coverage for categorical PPO with a fixed activity-ID tail."""
from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts import train_ppo_gpu as ppo
from rcpsp_bb_rl.data.parsing import load_instance
from rcpsp_bb_rl.ml.action_order import resolve_activity_order
from rcpsp_bb_rl.ml.models import BranchingTransformer, SeparateActorCritic
from rcpsp_bb_rl.ml.rl import BranchingEnv


class SingleActionPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_likelihood_entropy_and_gradient_credit_only_the_first_choice(self) -> None:
        logits = torch.tensor(
            [[math.log(0.8), math.log(0.1), math.log(0.1), 100.0]],
            dtype=torch.float64, requires_grad=True,
        )
        mask = torch.tensor([[True, True, True, False]])
        log_prob, entropy = ppo.compute_log_probs_entropy(logits, torch.tensor([0]), mask)

        self.assertAlmostEqual(log_prob.item(), math.log(0.8))
        self.assertAlmostEqual(entropy.item(), -sum(p * math.log(p) for p in (0.8, 0.1, 0.1)))
        log_prob.sum().backward()
        torch.testing.assert_close(
            logits.grad, torch.tensor([[0.2, -0.1, -0.1, 0.0]], dtype=torch.float64),
        )

        uniform_lp, uniform_entropy = ppo.compute_log_probs_entropy(
            torch.zeros(1, 3), torch.tensor([1]), torch.ones(1, 3, dtype=torch.bool),
        )
        self.assertAlmostEqual(uniform_lp.item(), -math.log(3), places=6)
        self.assertAlmostEqual(uniform_entropy.item(), math.log(3), places=6)

    def test_ppo_ratio_cannot_compensate_for_first_choice_with_tail_changes(self) -> None:
        mask = torch.ones(1, 3, dtype=torch.bool)
        action = torch.tensor([0])
        old_lp, _ = ppo.compute_log_probs_entropy(
            torch.tensor([[0.8, 0.1, 0.1]]).log(), action, mask,
        )
        new_lp, _ = ppo.compute_log_probs_entropy(
            torch.tensor([[0.72, 0.168, 0.112]]).log(), action, mask,
        )
        self.assertAlmostEqual(torch.exp(new_lp - old_lp).item(), 0.9, places=6)

    def test_unbatched_collection_matches_padded_replay_for_both_feature_schemas(self) -> None:
        device = torch.device("cpu")
        for resource_set in (False, True):
            with self.subTest(resource_set=resource_set):
                model = BranchingTransformer(
                    global_dim=3, candidate_dim=4, critic_feature_dim=2,
                    d_model=8, n_heads=2, n_layers=1, ffn_dim=16, dropout=0.0,
                    resource_feature_dim=2 if resource_set else 0,
                    interaction_feature_dim=3 if resource_set else 0,
                )
                model = SeparateActorCritic.from_shared(model)
                actor = ppo.ActorCritic(model).eval()
                buffer = ppo.RolloutBuffer()
                entropies = []
                for count, resource_count in ((4, 2), (2, 4), (1, 1), (3, 3)):
                    mask = torch.ones(count, dtype=torch.bool)
                    if count > 2:
                        mask[1] = False
                    obs = {
                        "candidate_feats": torch.randn(count, 4),
                        "global_feats": torch.randn(3),
                        "critic_feats": torch.randn(2),
                        "action_mask": mask,
                    }
                    if resource_set:
                        obs.update(
                            resource_feats=torch.randn(resource_count, 2),
                            candidate_resource_feats=torch.randn(count, resource_count, 3),
                            resource_mask=torch.ones(resource_count, dtype=torch.bool),
                        )
                    with torch.no_grad():
                        action, log_prob, entropy, value = actor.get_action_and_value(obs, device)
                    self.assertTrue(mask[action.item()].item())
                    buffer.add(
                        obs, action.item(), log_prob.item(), value.item(), False, False,
                        feasible_count=int(mask.sum()),
                    )
                    entropies.append(entropy.item())

                cand, glob, mask, critic, pad, _, resources, interactions, resource_mask = (
                    ppo.batch_observations(buffer.obs, device)
                )
                logits, values = model.forward_batch(
                    cand, glob, mask, critic, pad, resources, interactions, resource_mask,
                )
                replay_lp, replay_entropy = ppo.compute_log_probs_entropy(
                    logits, torch.tensor(buffer.actions), mask,
                )
                torch.testing.assert_close(replay_lp, torch.tensor(buffer.log_probs), atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(replay_entropy, torch.tensor(entropies), atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(values, torch.tensor(buffer.values), atol=1e-6, rtol=1e-5)
                self.assertEqual(replay_lp[2].item(), 0.0)
                self.assertEqual(replay_entropy[2].item(), 0.0)

                # The replay must retain gradients through the policy and resource encoder.
                loss = -(replay_lp * torch.tensor([1.0, -1.0, 0.0, 0.5])).mean()
                loss.backward()
                self.assertGreater(actor.policy_model.score_head.weight.grad.abs().sum().item(), 0.0)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all().item())
                if resource_set:
                    self.assertGreater(actor.policy_model.resource_proj.weight.grad.abs().sum().item(), 0.0)

    def test_default_tail_is_sorted_by_activity_id_even_for_unsorted_input(self) -> None:
        self.assertEqual(resolve_activity_order([30, 10, 20, 40], 2), [20, 10, 30, 40])

    def test_environment_sends_selected_first_then_id_tail_to_solver(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.rcp"
            path.write_text("5 1\n1\n0 0 3 2 3 4\n2 1 1 5\n3 1 1 5\n1 1 1 5\n0 0 0\n")
            env = BranchingEnv(load_instance(path), time_limit_s=5, dominance=False)
            obs = env.reset()
            sent_orders = []

            def record_order(*args, **kwargs):
                order = resolve_activity_order(*args, **kwargs)
                sent_orders.append(order)
                return order

            with patch("rcpsp_bb_rl.ml.rl.env.resolve_activity_order", side_effect=record_order):
                # Select the largest activity ID each time, then let DFS finish.
                while True:
                    step = env.step(len(obs["ready_ids"]) - 1)
                    if step.done:
                        break
                    obs = step.observation

            self.assertIn([4, 2, 3], sent_orders)
            self.assertEqual(env.result.best_makespan, 6)
            self.assertEqual(env.result.done_reason, "search_exhausted")

    def test_forced_and_all_infeasible_states_are_excluded_from_updates(self) -> None:
        episode = ppo.StagedEpisode(feasible_counts=[0, 1, 2, 3])
        indices, report = ppo.subsample_episode(
            episode, [True] * 4, cap=None, n_activities=5, time_bands=3,
            depth_bands=4, incumbent_window=1, rng=ppo.random.Random(42),
        )
        self.assertEqual(indices, [2, 3])
        self.assertEqual(report.n_forced_dropped, 2)

    def test_small_training_run_preserves_resource_batching_normalization_and_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instances = root / "instances"
            instances.mkdir()
            (instances / "one_resource.rcp").write_text(
                "5 1\n1\n0 0 3 2 3 4\n2 1 1 5\n3 1 1 5\n1 1 1 5\n0 0 0\n"
            )
            (instances / "two_resources.rcp").write_text(
                "5 2\n1 2\n0 0 0 3 2 3 4\n2 1 1 1 5\n3 1 1 1 5\n1 1 1 1 5\n0 0 0 0\n"
            )
            baseline = root / "optimal.json"
            baseline.write_text(json.dumps({"instances": {
                name: {"makespan": 6} for name in ("one_resource", "two_resources")
            }}))
            save_path = root / "output" / "policy.pt"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "root": str(instances), "pattern": "*.rcp", "resource_encoder": "set",
                "bc_checkpoint": None, "d_model": 8, "n_heads": 2, "n_layers": 1,
                "ffn_dim": 16, "dropout": 0.0, "total_env_steps": 40,
                "min_batch_size": 4, "min_episodes": 2, "episode_transition_cap": 2,
                "target_mb_size": 4, "ppo_epochs": 2, "time_limit_s": 5,
                "dominance": False, "tree_gamma_cost": 0.85, "tree_gamma_bonus": 1.0,
                "estimator_path": None, "oracle_node_counts_path": None,
                "eval_root": str(instances), "eval_pattern": "*.rcp",
                "eval_optimal_json": str(baseline), "eval_every_updates": 1,
                "eval_time_limit_s": 5, "save_path": str(save_path),
                "checkpoint_dir": str(root / "checkpoints"), "tensorboard": False,
                "device": "cpu", "seed": 42,
            }))
            with patch.object(sys, "argv", ["train_ppo_gpu.py", "--config", str(config_path)]):
                with contextlib.redirect_stdout(io.StringIO()):
                    ppo.main()

            checkpoint = torch.load(
                save_path.with_name("policy_best.pt"), map_location="cpu", weights_only=False,
            )
            self.assertGreater(checkpoint["update"], 0)
            self.assertGreater(checkpoint["value_norm"]["count"], 0)
            self.assertTrue(math.isfinite(checkpoint["value_norm"]["mean"]))
            self.assertEqual(checkpoint["eval_metrics"]["solved_frac"], 1.0)
            self.assertEqual(checkpoint["actor_critic_architecture"], "separate_transformers_v1")
            self.assertIn("critic_state", checkpoint)
            self.assertTrue(checkpoint["config"]["policy_only"])
            self.assertTrue(checkpoint["critic_config"]["value_only"])
            self.assertTrue(list((root / "checkpoints").glob("policy_ppo_step*.pt")))


if __name__ == "__main__":
    unittest.main()
