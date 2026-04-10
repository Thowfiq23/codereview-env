import os
import re
import uuid
import platform
import subprocess
import shutil
from subprocess import TimeoutExpired
from typing import Dict, Any, Tuple, Optional

from models import AgentAction, CodeObservation, ReviewState
from tasks import TASKS, GRADERS

# Configurable via env var — defaults to /tmp for Linux containers.
# Set WORKSPACE_BASE to a writable path on non-Linux hosts.
WORKSPACE_BASE = os.getenv("WORKSPACE_BASE", "/tmp/codereview_workspaces")

MAX_STEPS = 10

# Minimal clean environment passed to all subprocesses.
# Deliberately excludes server credentials (HF_TOKEN, API keys, etc.)
_CLEAN_ENV_KEYS = {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER", "LOGNAME"}

# Windows requires these vars for Python/subprocess to initialise correctly.
_WINDOWS_ENV_KEYS = {"SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "COMSPEC"}


def _make_clean_env(pythonpath: str) -> Dict[str, str]:
    """Return a minimal subprocess environment — no server credentials leaked."""
    keys = _CLEAN_ENV_KEYS.copy()
    if platform.system() == "Windows":
        keys |= _WINDOWS_ENV_KEYS
    clean = {k: v for k, v in os.environ.items() if k in keys}
    clean["PYTHONPATH"] = pythonpath
    clean["PYTHONDONTWRITEBYTECODE"] = "1"
    if "PATH" not in clean:
        clean["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    return clean


class CodeReviewEnvironment:
    def __init__(self):
        if os.path.isdir(WORKSPACE_BASE):
            shutil.rmtree(WORKSPACE_BASE, ignore_errors=True)
        os.makedirs(WORKSPACE_BASE, exist_ok=True)

        self.state = ReviewState(
            episode_id=str(uuid.uuid4()),
            task_id="",
            step_count=0,
            current_task_index=0,
            total_reward=0.0,
            done=False
        )
        self.current_task_data = None
        self._is_first_reset = True
        self.workspace_dir = ""
        # Tracks the highest pass rate reached this episode for delta-based rewards.
        # Only updated on improvement — prevents reward farming by repeated patches.
        self.prev_pass_rate: float = 0.0
        # Original test file contents keyed by relative path.
        # Used by _enforce_test_integrity() to restore files after every
        # execute_command — chmod 444 alone is bypassable via the shell.
        self._test_file_originals: Dict[str, str] = {}

    def _setup_workspace(self):
        """Create a fresh isolated workspace for this episode."""
        if self.workspace_dir and os.path.exists(self.workspace_dir):
            shutil.rmtree(self.workspace_dir, ignore_errors=True)

        self.workspace_dir = os.path.join(WORKSPACE_BASE, self.state.episode_id)
        os.makedirs(self.workspace_dir, exist_ok=True)

        self._test_file_originals = {}
        repo = self.current_task_data["repository"]
        for filepath, content in repo.items():
            full_path = os.path.join(self.workspace_dir, filepath)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            if filepath.startswith("tests/"):
                # Make read-only AND store original content.
                # chmod 444 alone is bypassable via shell ('chmod +w …').
                # _enforce_test_integrity() restores from this dict after
                # every execute_command, closing that shell bypass.
                os.chmod(full_path, 0o444)
                self._test_file_originals[filepath] = content

    def cleanup(self):
        """Explicitly remove the current workspace. Called on server shutdown."""
        if self.workspace_dir and os.path.exists(self.workspace_dir):
            shutil.rmtree(self.workspace_dir, ignore_errors=True)

    def _enforce_test_integrity(self) -> None:
        """
        Restore any test files modified or deleted by a shell command.

        chmod 444 only prevents direct writes through open(); an agent using
        execute_command can still run 'chmod +w tests/...' followed by an
        echo redirect or a Python one-liner to overwrite the oracle.  This
        method compares every test file against its original content and
        rewrites + re-locks any that differ, closing that bypass completely.
        Called unconditionally after every execute_command.
        """
        for relpath, original in self._test_file_originals.items():
            full_path = os.path.join(self.workspace_dir, relpath)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    current = f.read()
                if current == original:
                    continue  # untouched — fast path
            except OSError:
                current = None  # file was deleted

            # Restore: briefly make writable, rewrite, re-lock
            try:
                os.chmod(full_path, 0o644)
            except OSError:
                pass
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(original)
            os.chmod(full_path, 0o444)

        # Remove extra files created under tests/ that were not present at
        # episode start.  An agent can write tests/conftest.py via the shell
        # to inject fixtures that make broken tests appear to pass.
        tests_dir = os.path.join(self.workspace_dir, "tests")
        if os.path.isdir(tests_dir):
            tracked_abs = {
                os.path.normpath(os.path.join(self.workspace_dir, p))
                for p in self._test_file_originals
            }
            for dirpath, _, filenames in os.walk(tests_dir):
                for fname in filenames:
                    fpath = os.path.normpath(os.path.join(dirpath, fname))
                    if fpath not in tracked_abs:
                        try:
                            os.remove(fpath)
                        except OSError:
                            pass

    def reset(self, task_id: Optional[str] = None) -> CodeObservation:
        """
        Start a new episode.

        Parameters
        ----------
        task_id : str, optional
            Explicit task to load (e.g. "task_3_pr").  When provided the
            internal round-robin counter is overridden.  When omitted the
            environment cycles through tasks in order.
        """
        self.state.episode_id = str(uuid.uuid4())
        self.state.step_count = 0
        self.state.done = False
        self.state.total_reward = 0.0
        self.prev_pass_rate = 0.0

        task_ids = [t["task_id"] for t in TASKS]

        if task_id is not None and task_id in task_ids:
            # Caller requested a specific task — honour it exactly.
            self.state.current_task_index = task_ids.index(task_id)
        elif not self._is_first_reset:
            # Default: advance round-robin.
            self.state.current_task_index = (self.state.current_task_index + 1) % len(TASKS)
        self._is_first_reset = False

        self.current_task_data = TASKS[self.state.current_task_index]
        self.state.task_id = self.current_task_data["task_id"]

        self._setup_workspace()

        available_files = list(self.current_task_data["repository"].keys())

        return CodeObservation(
            task_id=self.state.task_id,
            context=self.current_task_data["description"],
            available_files=available_files,
            action_result="Sandbox ready. Run 'pytest tests/' to see what is failing.",
            step_number=self.state.step_count,
            done=False,
            reward=0.0
        )

    def _get_test_pass_rate(self) -> float:
        """
        Dispatch to the per-task grader and return pass rate [0.0, 1.0].

        Uses the GRADERS dict from tasks.py so each task has an explicit,
        named grader function — not a single shared anonymous pytest call.
        The grader runs pytest internally with a clean minimal environment.
        """
        grader = GRADERS.get(self.state.task_id)
        if grader is None:
            return 0.0
        return grader(self.workspace_dir)

    def step(self, action_obj: AgentAction) -> Tuple[CodeObservation, float, bool, Dict[str, Any]]:
        # Guard: /reset must be called before the first /step
        if self.current_task_data is None:
            obs = CodeObservation(
                task_id="none",
                context="No active episode.",
                available_files=[],
                action_result="Call POST /reset before calling /step.",
                step_number=0,
                done=False,
                reward=0.0
            )
            return obs, 0.0, False, {"error": "no_active_episode_call_reset_first"}

        # If episode already ended, return a clean terminal observation.
        if self.state.done:
            obs = CodeObservation(
                task_id=self.state.task_id,
                context=self.current_task_data["description"] if self.current_task_data else "",
                available_files=list(self.current_task_data["repository"].keys()) if self.current_task_data else [],
                action_result="Episode is already done. Call /reset to start a new episode.",
                step_number=self.state.step_count,
                done=True,
                reward=0.0
            )
            return obs, 0.0, True, {"error": "episode_already_done"}

        try:
            self.state.step_count += 1
            available_files = list(self.current_task_data["repository"].keys())

            obs_result = ""
            reward = 0.0
            done = False
            info = {}

            env = _make_clean_env(self.workspace_dir)

            # ----------------------------------------------------------------
            # TOOL 1: EXECUTE COMMAND
            # ----------------------------------------------------------------
            if action_obj.action_type == "execute_command":
                cmd = action_obj.command
                if not cmd:
                    obs_result = "Error: No bash command provided."
                    reward = 0.0
                else:
                    try:
                        result = subprocess.run(
                            cmd, cwd=self.workspace_dir, shell=True,
                            capture_output=True, text=True, timeout=15, env=env,
                            start_new_session=True,
                        )
                        output = result.stdout if result.stdout else result.stderr
                        MAX_OUTPUT_CHARS = 4000
                        if len(output) > MAX_OUTPUT_CHARS:
                            output = (
                                f"...[TRUNCATED — {len(output)} chars, "
                                f"showing last {MAX_OUTPUT_CHARS}]...\n"
                                + output[-MAX_OUTPUT_CHARS:]
                            )
                        obs_result = f"--- BASH OUTPUT (Exit Code {result.returncode}) ---\n{output}"
                        # Successful commands (exit 0) give a small exploration signal.
                        # Failed commands (non-zero exit) give a smaller signal —
                        # this differentiates useful exploration from wasted steps.
                        reward = 0.01 if result.returncode == 0 else 0.005
                    except TimeoutExpired:
                        obs_result = "Error: Command timed out after 15 seconds. Avoid long-running or interactive commands."
                        reward = 0.005  # timeout is a wasted step

                # Restore any test files the shell may have tampered with.
                # chmod 444 alone is bypassable via 'chmod +w'; this closes it.
                self._enforce_test_integrity()

            # ----------------------------------------------------------------
            # TOOL 2: PATCH FILE — Delta-based dense reward
            # ----------------------------------------------------------------
            elif action_obj.action_type == "patch_file":
                target = action_obj.target_file
                new_content = action_obj.new_content
                if not target or not new_content:
                    obs_result = "Error: target_file and new_content are required."
                    reward = 0.0
                else:
                    workspace_real = os.path.realpath(self.workspace_dir)
                    full_path = os.path.realpath(
                        os.path.join(self.workspace_dir, target)
                    )
                    tests_real = os.path.realpath(
                        os.path.join(self.workspace_dir, "tests")
                    )

                    if not full_path.startswith(workspace_real + os.sep):
                        obs_result = "Security Error: Path traversal detected and blocked."
                        # Explicit penalty — destructive intent, not neutral exploration.
                        reward = -0.05

                    elif full_path.startswith(tests_real + os.sep) or full_path == tests_real:
                        obs_result = "Security Error: Test files are read-only and cannot be modified by the agent."
                        # Explicit penalty — attempt to tamper with the oracle.
                        reward = -0.05

                    else:
                        os.makedirs(os.path.dirname(full_path), exist_ok=True)
                        import py_compile, tempfile
                        if full_path.endswith(".py"):
                            try:
                                with tempfile.NamedTemporaryFile(
                                    mode="w", suffix=".py", delete=False,
                                    encoding="utf-8"
                                ) as tmp:
                                    tmp.write(new_content)
                                    tmp_path = tmp.name
                                py_compile.compile(tmp_path, doraise=True)
                                os.unlink(tmp_path)
                            except py_compile.PyCompileError as e:
                                os.unlink(tmp_path)
                                obs_result = f"Syntax Error in patch: {e}. File was NOT written. Fix the syntax and try again."
                                reward = 0.005  # syntax error — wasted step, slight penalty
                                full_path = None

                        if full_path is not None and not obs_result:
                            with open(full_path, "w", encoding="utf-8") as f:
                                f.write(new_content)
                            obs_result = f"Successfully patched: {target}"

                            # Delta-based reward: only reward genuine test improvements.
                            # raw  = current pass rate after this patch.
                            # delta = improvement over the best pass rate so far.
                            # If the agent re-patches with no new test gains, reward is
                            # near-zero — no farming possible.
                            raw = self._get_test_pass_rate()
                            delta = round(raw - self.prev_pass_rate, 4)

                            if delta > 0:
                                # Genuine progress: reward proportional to improvement.
                                # Range: (0.01, 0.89] — capped below 1.0 to force submit.
                                reward = round(0.01 + delta * 0.88, 4)
                                self.prev_pass_rate = raw   # advance the baseline
                            else:
                                # No improvement (stall or regression) — minimal signal.
                                reward = 0.005

                            # Expose actual pass rate for accurate breakdown metadata.
                            info["test_pass_rate"] = round(raw, 4)
                            self.state.total_reward = round(
                                max(self.state.total_reward, reward), 4
                            )

            # ----------------------------------------------------------------
            # TOOL 3: SUBMIT REVIEW — Final execution grader
            # ----------------------------------------------------------------
            elif action_obj.action_type == "submit_review":
                try:
                    result = subprocess.run(
                        "pytest tests/ -v",
                        cwd=self.workspace_dir, shell=True,
                        capture_output=True, text=True, timeout=15, env=env
                    )
                    if result.returncode == 0:
                        reward = 0.99
                        feedback = "SUCCESS: All tests passed."
                        info["test_pass_rate"] = 1.0
                    else:
                        raw = self._get_test_pass_rate()
                        # Map to (0.01, 0.98) — open interval
                        reward = round(0.01 + raw * 0.97, 4)
                        feedback = f"FAILED: Tests still failing.\n{result.stdout[-500:]}"
                        info["test_pass_rate"] = round(raw, 4)
                except TimeoutExpired:
                    reward = 0.01
                    feedback = "FAILED: Test suite timed out."
                    info["test_pass_rate"] = 0.0

                obs_result = f"Evaluation complete. {feedback}"
                done = True
                self.state.total_reward = max(self.state.total_reward, reward)
                info["feedback"] = feedback
                shutil.rmtree(self.workspace_dir, ignore_errors=True)

            # Server-side max_steps cutoff
            if self.state.step_count >= MAX_STEPS and not done:
                done = True
                obs_result += f"\n[System] Max steps ({MAX_STEPS}) reached. Episode terminated."
                shutil.rmtree(self.workspace_dir, ignore_errors=True)

            self.state.done = done

            obs = CodeObservation(
                task_id=self.state.task_id,
                context=self.current_task_data["description"],
                available_files=available_files,
                action_result=obs_result,
                step_number=self.state.step_count,
                done=done,
                reward=reward
            )
            return obs, reward, done, info

        except Exception as e:
            current_done = self.state.done
            if self.workspace_dir and os.path.exists(self.workspace_dir):
                if current_done:
                    shutil.rmtree(self.workspace_dir, ignore_errors=True)
            obs = CodeObservation(
                task_id=self.state.task_id if self.current_task_data else "error",
                context="Error state",
                available_files=[],
                action_result=f"Internal Error: {str(e)}",
                step_number=self.state.step_count,
                done=current_done,
                reward=0.0
            )
            return obs, 0.0, current_done, {"error": str(e)}
