from researchclaw.fsm.states import State
from researchclaw.models import TrialMeta
from researchclaw.config import ResearchClawConfig


class TestState:
    def test_all_states_exist(self) -> None:
        expected = {
            "EXPERIMENT_PLAN",
            "EXPERIMENT_IMPLEMENT",
            "EXPERIMENT_EXECUTE",
            "EVAL_IMPLEMENT",
            "EVAL_EXECUTE",
            "EXPERIMENT_REPORT",
            "DECIDE",
            "VIEW_SUMMARY",
            "SETTINGS",
            "MERGE_LOOP",
        }
        assert {s.name for s in State} == expected

    def test_state_values_are_lowercase(self) -> None:
        for s in State:
            assert s.value == s.name.lower()

    def test_state_is_string(self) -> None:
        assert State.DECIDE == "decide"
        assert isinstance(State.DECIDE, str)


class TestTrialMeta:
    def test_default_instantiation(self) -> None:
        meta = TrialMeta()
        assert meta.trial_number == 1
        assert meta.status == "pending"
        assert meta.state == "experiment_plan"
        assert meta.plan_approved_at is None
        assert meta.experiment_exit_code is None
        assert meta.eval_exit_code is None
        assert meta.decision is None
        assert meta.decision_reasoning is None

    def test_custom_instantiation(self) -> None:
        meta = TrialMeta(
            trial_number=3,
            status="running",
            state="experiment_execute",
            created_at="2026-03-01T10:00:00+00:00",
            updated_at="2026-03-01T10:35:00+00:00",
            plan_approved_at="2026-03-01T10:05:00+00:00",
            experiment_exit_code=0,
            eval_exit_code=None,
            decision="next_trial",
            decision_reasoning="Results promising, try different hyperparameters",
        )
        assert meta.trial_number == 3
        assert meta.status == "running"
        assert meta.state == "experiment_execute"
        assert meta.experiment_exit_code == 0
        assert meta.decision == "next_trial"

    def test_timestamps_auto_generated(self) -> None:
        meta = TrialMeta()
        assert meta.created_at is not None
        assert meta.updated_at is not None
        assert len(meta.created_at) > 0
        assert len(meta.updated_at) > 0


class TestResearchClawConfig:
    def test_default_values(self) -> None:
        config = ResearchClawConfig()
        assert config.model == "claude-opus-4-6"
        assert config.max_trials == 20
        assert config.max_retries == 5
        assert config.autopilot is False
        assert config.experiment_timeout_seconds == 3600
        assert config.python_command == "python3"

    def test_custom_values(self) -> None:
        config = ResearchClawConfig(
            model="gpt-4",
            max_trials=10,
            max_retries=3,
            autopilot=True,
            experiment_timeout_seconds=7200,
            python_command="python3.12",
        )
        assert config.model == "gpt-4"
        assert config.max_trials == 10
        assert config.max_retries == 3
        assert config.autopilot is True
        assert config.experiment_timeout_seconds == 7200
        assert config.python_command == "python3.12"
