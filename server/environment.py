import os
import re
import uuid
import platform
import subprocess
import shutil
from subprocess import TimeoutExpired
from typing import Dict, Any, Tuple

from models import AgentAction, CodeObservation, ReviewState
from tasks import TASKS

WORKSPACE_BASE = "/tmp/codereview_workspaces"
MAX_STEPS = 10

# Minimal clean environment passed to all subprocesses.
# Deliberately excludes server credentials (HF_TOKEN, API keys, etc.)
_CLEAN_ENV_KEYS = {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER", "LOGNAME"}

# Windows requires these vars for Python/subprocess to initialise correctly.
# Without SYSTEMROOT Python raises: Fatal Python error: _Py_HashRandomization_Init
_WINDOWS_ENV_KEYS = {"SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "COMSPEC"}


def _make_clean_env(pythonpath: str) -> Dict[str, str]:
    """Return a minimal subprocess environment — no server credentials leaked.

    On Windows the standard allowlist is extended with OS-required variables
    so that Python and pytest can initialise inside the sandbox.
    """
    keys = _CLEAN_ENV_KEYS.copy()
    if platform.system() == "Windows":
        keys |= _WINDOWS_ENV_KEYS
    clean = {k: v for k, v in os.environ.items() if k in keys}
    clean["PYTHONPATH"] = pythonpath
    # Ensure python/pytest are findable even in minimal PATH
    if "PATH" not in clean:
        clean["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    return clean


class CodeReviewEnvironment:
    def __init__(self):
        # Purge any workspace directories left over from a previous server run
        # (orphaned UUID dirs in /tmp/codereview_workspaces/ that the new
        # instance knows nothing about).
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

    def _setup_workspace(self):
        """Create a fresh isolated workspace for this episode, cleaning up the old one."""
        # Clean up previous workspace to prevent disk fill
        if self.workspace_dir and os.path.exists(self.workspace_dir):
            shutil.rmtree(self.workspace_dir, ignore_errors=True)

        self.workspace_dir = os.path.join(WORKSPACE_BASE, self.state.episode_id)
        os.makedirs(self.workspace_dir, exist_ok=True)

        repo = self.current_task_data["repository"]
        for filepath, content in repo.items():
            full_path = os.path.join(self.workspace_dir, filepath)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content)
            # Make test files read-only so the agent cannot overwrite them
            # and earn a free 1.0 by replacing the grader with a trivial pass.
            if filepath.startswith("tests/"):
                os.chmod(full_path, 0o444)

    def cleanup(self):
        """Explicitly remove the current workspace. Called on server shutdown."""
        if self.workspace_dir and os.path.exists(self.workspace_dir):
            shutil.rmtree(self.workspace_dir, ignore_errors=True)

    def reset(self) -> CodeObservation:
        self.state.episode_id = str(uuid.uuid4())
        self.state.step_count = 0
        self.state.done = False
        self.state.total_reward = 0.0

        if not self._is_first_reset:
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
            reward=0.01
        )

    def _get_test_pass_rate(self) -> float:
        """Run pytest silently and return the fraction of tests passing (0.0–1.0)."""
        env = _make_clean_env(self.workspace_dir)
        try:
            result = subprocess.run(
                "pytest tests/ --disable-warnings -v",
                cwd=self.workspace_dir, shell=True,
                capture_output=True, text=True, timeout=15, env=env
            )
        except TimeoutExpired:
            return 0.0

        if result.returncode == 0:
            return 1.0

        output = result.stdout + result.stderr
        n_passed = sum(1 for line in output.splitlines() if " PASSED" in line)
        n_failed = sum(1 for line in output.splitlines() if " FAILED" in line or " ERROR" in line)
        total = n_passed + n_failed
        if total > 0:
            return n_passed / total

        # Fallback: parse summary line e.g. "1 failed, 1 passed in 0.12s"
        m_pass = re.search(r"(\d+) passed", output)
        m_fail = re.search(r"(\d+) failed", output)
        p = int(m_pass.group(1)) if m_pass else 0
        f = int(m_fail.group(1)) if m_fail else 0
        return p / (p + f) if (p + f) > 0 else 0.0

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
                reward=0.01
            )
            return obs, 0.01, False, {"error": "no_active_episode_call_reset_first"}

        # If episode already ended, return a clean terminal observation — do not raise
        if self.state.done:
            obs = CodeObservation(
                task_id=self.state.task_id,
                context=self.current_task_data["description"] if self.current_task_data else "",
                available_files=list(self.current_task_data["repository"].keys()) if self.current_task_data else [],
                action_result="Episode is already done. Call /reset to start a new episode.",
                step_number=self.state.step_count,
                done=True,
                reward=0.01
            )
            return obs, 0.01, True, {"error": "episode_already_done"}

        try:
            self.state.step_count += 1
            available_files = list(self.current_task_data["repository"].keys())

            obs_result = ""
            reward = 0.01  # floor: no action path should return exactly 0.0
            done = False
            info = {}

            env = _make_clean_env(self.workspace_dir)

            # --- TOOL 1: EXECUTE COMMAND (Real Bash Terminal) ---
            if action_obj.action_type == "execute_command":
                cmd = action_obj.command
                if not cmd:
                    obs_result = "Error: No bash command provided."
                else:
                    try:
                        result = subprocess.run(
                            cmd, cwd=self.workspace_dir, shell=True,
                            capture_output=True, text=True, timeout=15, env=env,
                            # New session isolates the child's process group so
                            # signals sent inside the sandbox (e.g. kill -9 -1)
                            # cannot escape to kill the FastAPI server process.
                            start_new_session=True,
                        )
                        output = result.stdout if result.stdout else result.stderr
                        # Truncate to prevent context-window exhaustion on the LLM side.
                        # Verbose commands (npm install, find /, recursive pytest) can
                        # emit megabytes; keep the last 4000 chars which contain the
                        # actionable failure lines.
                        MAX_OUTPUT_CHARS = 4000
                        if len(output) > MAX_OUTPUT_CHARS:
                            output = (
                                f"...[TRUNCATED — {len(output)} chars, "
                                f"showing last {MAX_OUTPUT_CHARS}]...\n"
                                + output[-MAX_OUTPUT_CHARS:]
                            )
                        obs_result = f"--- BASH OUTPUT (Exit Code {result.returncode}) ---\n{output}"
                    except TimeoutExpired:
                        obs_result = "Error: Command timed out after 15 seconds. Avoid long-running or interactive commands."
                # execute_command never returns exactly 0 — evaluator requires open interval (0, 1)
                reward = 0.01

            # --- TOOL 2: PATCH FILE (Physical Disk Write + Dense Partial Reward) ---
            elif action_obj.action_type == "patch_file":
                target = action_obj.target_file
                new_content = action_obj.new_content
                if not target or not new_content:
                    obs_result = "Error: target_file and new_content are required."
                else:
                    # Resolve both the target path and the tests directory to their
                    # canonical real paths. String-prefix checks (target.startswith)
                    # are bypassable via traversal like "auth/../tests/test_auth.py".
                    workspace_real = os.path.realpath(self.workspace_dir)
                    full_path = os.path.realpath(
                        os.path.join(self.workspace_dir, target)
                    )
                    tests_real = os.path.realpath(
                        os.path.join(self.workspace_dir, "tests")
                    )

                    if not full_path.startswith(workspace_real + os.sep):
                        obs_result = "Security Error: Path traversal detected and blocked."
                    elif full_path.startswith(tests_real + os.sep) or full_path == tests_real:
                        # Block writes to the tests/ directory regardless of how the
                        # path was constructed (direct or via traversal).
                        obs_result = "Security Error: Test files are read-only and cannot be modified by the agent."
                    else:
                        os.makedirs(os.path.dirname(full_path), exist_ok=True)
                        # Quick syntax check before committing the file — gives the
                        # agent immediate feedback instead of a cryptic reward=0.0.
                        import py_compile, tempfile
                        if full_path.endswith(".py"):
                            try:
                                with tempfile.NamedTemporaryFile(
                                    mode="w", suffix=".py", delete=False
                                ) as tmp:
                                    tmp.write(new_content)
                                    tmp_path = tmp.name
                                py_compile.compile(tmp_path, doraise=True)
                                os.unlink(tmp_path)
                            except py_compile.PyCompileError as e:
                                os.unlink(tmp_path)
                                obs_result = f"Syntax Error in patch: {e}. File was NOT written. Fix the syntax and try again."
                                # Skip the rest — don't write a broken file
                                full_path = None
                        if full_path is not None and not obs_result:
                            with open(full_path, "w") as f:
                                f.write(new_content)
                            obs_result = f"Successfully patched: {target}"
                            # Dense reward: strictly in (0, 1) — evaluator requires open interval.
                            # Map test_pass_rate to (0.01, 0.89): partial credit without hitting 0 or 1.
                            raw = self._get_test_pass_rate()
                            reward = round(0.01 + raw * 0.88, 4)
                            # Accumulate to total_reward so state() reflects trajectory progress
                            self.state.total_reward = round(
                                max(self.state.total_reward, reward), 4
                            )

            # --- TOOL 3: SUBMIT REVIEW (Final Execution Grader) ---
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
                    else:
                        raw = self._get_test_pass_rate()
                        # Map to (0.01, 0.98) — never exactly 0 or 1 (evaluator requires open interval)
                        reward = round(0.01 + raw * 0.97, 4)
                        feedback = f"FAILED: Tests still failing.\n{result.stdout[-500:]}"
                except TimeoutExpired:
                    reward = 0.01
                    feedback = "FAILED: Test suite timed out."

                obs_result = f"Evaluation complete. {feedback}"
                done = True
                self.state.total_reward = max(self.state.total_reward, reward)
                info["feedback"] = feedback
                # Cleanup workspace after episode ends
                shutil.rmtree(self.workspace_dir, ignore_errors=True)

            # Server-side max_steps cutoff — prevents runaway episodes
            if self.state.step_count >= MAX_STEPS and not done:
                done = True
                obs_result += f"\n[System] Max steps ({MAX_STEPS}) reached. Episode terminated."
                # Clean up workspace — don't wait for the next reset()
                shutil.rmtree(self.workspace_dir, ignore_errors=True)

            # Commit the done flag to state BEFORE building the observation so
            # that any exception thrown inside CodeObservation() or below is
            # caught by the except block with the correct terminal state.
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
            # self.state.done is authoritative — it is updated before obs
            # construction so this always reflects the correct terminal state
            # even when the exception fires mid-obs-build.
            current_done = self.state.done
            if self.workspace_dir and os.path.exists(self.workspace_dir):
                # Always clean up on any exception when the episode is done.
                # When not done the workspace must survive for the next step,
                # so only remove it if the episode has truly ended.
                if current_done:
                    shutil.rmtree(self.workspace_dir, ignore_errors=True)
            obs = CodeObservation(
                task_id=self.state.task_id if self.current_task_data else "error",
                context="Error state",
                available_files=[],
                action_result=f"Internal Error: {str(e)}",
                step_number=self.state.step_count,
                done=current_done,
                reward=0.01
            )
            return obs, 0.01, current_done, {"error": str(e)}
