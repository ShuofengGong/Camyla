"""
Python interpreter for executing code snippets and capturing their output.
Supports:
- captures stdout and stderr
- captures exceptions and stack traces
- limits execution time
"""

import logging
import os
import queue
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.queues import Queue as QueueType
from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict, Any

import humanize
from dataclasses_json import DataClassJsonMixin

# Check for psutil availability for enhanced process cleanup
try:
    import psutil
    HAS_PSUTIL = True
    logger = logging.getLogger("camyla")
    logger.debug("psutil available - enhanced process cleanup enabled")
except ImportError:
    HAS_PSUTIL = False
    logger = logging.getLogger("camyla")
    logger.info("psutil not available - using basic process cleanup")


# Note: We use TimeoutError for all timeout-related failures
# to ensure consistent handling by the experiment system


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """

    term_out: List[str]
    exec_time: float
    exc_type: Optional[str]
    exc_info: Optional[Dict] = None
    exc_stack: Optional[List[Tuple]] = None


def exception_summary(e, working_dir, exec_file_name, format_tb_ipython):
    """Generates a string that summarizes an exception and its stack trace (either in standard python repl or in IPython format)."""
    if format_tb_ipython:
        import IPython.core.ultratb

        tb = IPython.core.ultratb.VerboseTB(tb_offset=1, color_scheme="NoColor")
        tb_str = str(tb.text(*sys.exc_info()))
    else:
        tb_lines = traceback.format_exception(e)
        # skip parts of stack trace in weflow code
        tb_str = "".join(
            [l for l in tb_lines if "treesearch/" not in l and "importlib" not in l]
        )

    # replace whole path to file with just filename (to remove agent workspace dir)
    tb_str = tb_str.replace(str(working_dir / exec_file_name), exec_file_name)

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))

    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    return tb_str, e.__class__.__name__, exc_info, exc_stack


class RedirectQueue:
    def __init__(self, queue):
        self.queue = queue

    def write(self, msg):
        self.queue.put(msg)

    def flush(self):
        pass


def _run_session_worker(config: dict, code_inq, result_outq, event_outq) -> None:
    """
    Top-level worker function for spawn-based multiprocessing.
    
    This function runs in a separate process and handles code execution via subprocess.
    It only takes picklable arguments to be compatible with 'spawn' start method.
    
    Args:
        config: Dict with keys: working_dir, timeout, python_path, env_vars, agent_file_name
        code_inq: Queue for receiving code to execute
        result_outq: Queue for sending stdout/stderr output
        event_outq: Queue for sending execution events
    """
    import subprocess as sp
    import re
    import sys
    import traceback
    
    try:
        # Parse config
        working_dir = Path(config['working_dir'])
        timeout = config['timeout']
        python_path = config['python_path']
        env_vars = config.get('env_vars', {})
        agent_file_name = config['agent_file_name']
        
        # Validate working directory
        if not working_dir.exists():
            event_outq.put(("state:error", "WorkerStartupError", {"error": f"Working directory does not exist: {working_dir}"}, None))
            return
        
        # Setup environment
        for key, value in env_vars.items():
            os.environ[key] = value
        os.chdir(str(working_dir))
        
        # Send startup success signal
        event_outq.put(("state:worker_started",))

    except Exception as e:
        # Startup failed - send error and exit
        try:
            error_msg = f"Worker startup failed: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            event_outq.put(("state:error", "WorkerStartupError", {"error": error_msg}, None))
        except:
            pass  # If we can't even send the error, just exit
        return

    def parse_exception_type(stderr: str) -> str:
        """Parse exception type from stderr text."""
        if not stderr:
            return "RuntimeError"
        lines = stderr.strip().split('\n')
        for line in reversed(lines):
            match = re.match(r'^(\w+Error|\w+Exception|KeyboardInterrupt):', line)
            if match:
                return match.group(1)
        return "RuntimeError"

    # Main execution loop
    while True:
        try:
            code = code_inq.get()
            os.chdir(str(working_dir))
            
            # Write code to file
            script_path = working_dir / agent_file_name
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            try:
                # Prepare environment
                env = os.environ.copy()
                env.update(env_vars)
                # Add working directory to PYTHONPATH
                pythonpath = env.get('PYTHONPATH', '')
                env['PYTHONPATH'] = f"{working_dir}:{pythonpath}" if pythonpath else str(working_dir)
                
                # Use Popen with os.setsid to create a new process group.
                # This ensures os.killpg() can kill ALL descendants (DataLoader
                # workers, nnUNet sub-processes, etc.) in one shot, preventing
                # GPU memory leaks from orphaned child processes.
                proc = sp.Popen(
                    [python_path, str(script_path)],
                    cwd=str(working_dir),
                    stdout=sp.PIPE,
                    stderr=sp.PIPE,
                    text=True,
                    env=env,
                    preexec_fn=os.setsid,
                )
                
                # Send PID back so the parent Interpreter can killpg on cleanup
                event_outq.put(("state:ready", proc.pid))
                
                try:
                    stdout, stderr = proc.communicate(timeout=timeout)
                except sp.TimeoutExpired:
                    # Kill the entire process group (experiment + all children)
                    pgid = os.getpgid(proc.pid)
                    try:
                        os.killpg(pgid, signal.SIGTERM)
                    except (ProcessLookupError, OSError):
                        pass
                    import time as _time
                    _time.sleep(2)
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
                    # Read any remaining output after kill
                    try:
                        stdout, stderr = proc.communicate(timeout=5)
                    except sp.TimeoutExpired:
                        proc.kill()
                        stdout, stderr = proc.communicate(timeout=3)
                    except Exception:
                        stdout, stderr = "", ""
                    
                    result_outq.put(f"TimeoutError: Execution exceeded {timeout} seconds")
                    if stdout:
                        result_outq.put(stdout)
                    if stderr:
                        result_outq.put(stderr)
                    event_outq.put(("state:finished", "TimeoutError", {"timeout": timeout}, None))
                    result_outq.put("<|EOF|>")
                    continue
                
                # Normal completion: output stdout
                if stdout:
                    for line in stdout.strip().split('\n'):
                        if line:
                            result_outq.put(line)
                
                # Output stderr (contains traceback on error)
                if stderr:
                    for line in stderr.strip().split('\n'):
                        if line:
                            result_outq.put(line)
                
                # Check return code
                if proc.returncode != 0:
                    exc_type = parse_exception_type(stderr)
                    event_outq.put(("state:finished", exc_type, {"stderr": stderr}, None))
                else:
                    event_outq.put(("state:finished", None, None, None))
                    
            except Exception as e:
                error_msg = f"Execution error: {type(e).__name__}: {str(e)}"
                result_outq.put(error_msg)
                event_outq.put(("state:finished", type(e).__name__, {"error": str(e)}, None))

            # put EOF marker to indicate that we're done
            result_outq.put("<|EOF|>")
            
        except Exception as loop_error:
            # Error in main loop - try to report and continue
            try:
                error_msg = f"Worker loop error: {type(loop_error).__name__}: {str(loop_error)}"
                result_outq.put(error_msg)
                event_outq.put(("state:finished", type(loop_error).__name__, {"error": str(loop_error)}, None))
                result_outq.put("<|EOF|>")
            except:
                pass  # Can't report error, will crash


class Interpreter:
    def __init__(
        self,
        working_dir: Union[Path, str],
        timeout: int = 3600,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
        env_vars: Dict[str, str] = {},
        use_conda: bool = False,
        conda_env: str = "py310",
    ):
        """
        Simulates a standalone Python REPL with an execution time limit.
        Supports conda environment execution via subprocess.

        Args:
            working_dir (Union[Path, str]): working directory of the agent
            timeout (int, optional): Timeout for each code execution step. Defaults to 3600.
            format_tb_ipython (bool, optional): Whether to use IPython or default python REPL formatting for exceptions. Defaults to False.
            agent_file_name (str, optional): The name for the agent's code file. Defaults to "runfile.py".
            env_vars (Dict[str, str], optional): Environment variables to set in the child process. Defaults to {}.
            use_conda (bool, optional): Whether to use conda environment. Defaults to False.
            conda_env (str, optional): Name of the conda environment to use. Defaults to "py310".
        """
        # this really needs to be a path, otherwise causes issues that don't raise exc
        self.working_dir = Path(working_dir).resolve()
        assert (
            self.working_dir.exists()
        ), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.agent_file_name = agent_file_name
        self.process = None  # Will be set by create_process()
        self._experiment_pgid = None  # PID/PGID of the experiment subprocess (for killpg)
        self.env_vars = env_vars
        
        # Conda configuration
        self.use_conda = use_conda
        self.conda_env = conda_env
        self.python_path = self._find_python_interpreter()
        
        # Log configuration
        if self.use_conda:
            logger.info(f"🐍 Interpreter using conda env '{self.conda_env}': {self.python_path}")
        else:
            logger.info(f"🐍 Interpreter using system Python: {self.python_path}")
        
        # Warn if psutil is not available for enhanced process cleanup
        if not HAS_PSUTIL:
            logger.warning(
                "For better process cleanup, consider installing psutil: pip install psutil"
            )

    def _find_python_interpreter(self) -> str:
        """Find Python interpreter path based on conda configuration."""
        import subprocess as sp
        
        if not self.use_conda:
            return sys.executable
        
        try:
            # Method 1: Try using CONDA_EXE environment variable
            conda_exe = os.environ.get('CONDA_EXE')
            if conda_exe and os.path.exists(conda_exe):
                result = sp.run(
                    [conda_exe, 'run', '-n', self.conda_env, 'which', 'python'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    python_path = result.stdout.strip()
                    if python_path and os.path.exists(python_path):
                        logger.info(f"🔍 Found conda Python via CONDA_EXE: {python_path}")
                        return python_path
            
            # Method 2: Try common conda installation paths
            common_conda_paths = [
                '/opt/conda/bin/conda',
                os.path.expanduser('~/anaconda3/bin/conda'),
                os.path.expanduser('~/miniconda3/bin/conda'),
                '/usr/local/anaconda3/bin/conda',
            ]
            
            for conda_path in common_conda_paths:
                if os.path.exists(conda_path):
                    result = sp.run(
                        [conda_path, 'run', '-n', self.conda_env, 'which', 'python'],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        python_path = result.stdout.strip()
                        if python_path and os.path.exists(python_path):
                            logger.info(f"🔍 Found conda Python via {conda_path}: {python_path}")
                            return python_path
            
            # Method 3: Try direct path to conda environment
            direct_paths = [
                f"/opt/conda/envs/{self.conda_env}/bin/python",
                os.path.expanduser(f"~/anaconda3/envs/{self.conda_env}/bin/python"),
                os.path.expanduser(f"~/miniconda3/envs/{self.conda_env}/bin/python"),
            ]
            
            for direct_path in direct_paths:
                if os.path.exists(direct_path):
                    logger.info(f"🔍 Found conda Python at direct path: {direct_path}")
                    return direct_path
            
            logger.warning(f"⚠️ Could not find conda environment '{self.conda_env}', falling back to system Python")
            
        except Exception as e:
            logger.warning(f"⚠️ Error finding conda Python: {e}, falling back to system Python")
        
        return sys.executable

    def child_proc_setup(self, result_outq: Any) -> None:
        # disable all warnings (before importing anything)
        import shutup

        shutup.mute_warnings()

        for key, value in self.env_vars.items():
            os.environ[key] = value

        os.chdir(str(self.working_dir))

        # this seems to only  benecessary because we're exec'ing code from a string,
        # a .py file should be able to import modules from the cwd anyway
        sys.path.append(str(self.working_dir))

        # capture stdout and stderr
        # trunk-ignore(mypy/assignment)
        sys.stdout = sys.stderr = RedirectQueue(result_outq)

    def _run_session(
        self, code_inq: Any, result_outq: Any, event_outq: Any
    ) -> None:
        """DEPRECATED: This method is no longer used. See _run_session_worker() instead."""
        """Run code execution session using subprocess.
        
        This method uses subprocess.run() to execute code with the configured
        Python interpreter (conda or system), capturing stdout/stderr as text.
        """
        import subprocess as sp
        
        # Minimal setup for child process
        for key, value in self.env_vars.items():
            os.environ[key] = value
        os.chdir(str(self.working_dir))

        while True:
            code = code_inq.get()
            os.chdir(str(self.working_dir))
            
            # Write code to file
            script_path = self.working_dir / self.agent_file_name
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            event_outq.put(("state:ready",))
            
            try:
                # Prepare environment
                env = os.environ.copy()
                env.update(self.env_vars)
                # Add working directory to PYTHONPATH
                pythonpath = env.get('PYTHONPATH', '')
                env['PYTHONPATH'] = f"{self.working_dir}:{pythonpath}" if pythonpath else str(self.working_dir)
                
                # Execute using subprocess with configured Python interpreter
                result = sp.run(
                    [self.python_path, str(script_path)],
                    cwd=str(self.working_dir),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=env
                )
                
                # Output stdout
                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            result_outq.put(line)
                
                # Output stderr (contains traceback on error)
                if result.stderr:
                    for line in result.stderr.strip().split('\n'):
                        if line:
                            result_outq.put(line)
                
                # Check return code
                if result.returncode != 0:
                    # Parse exception type from stderr text
                    exc_type = self._parse_exception_type(result.stderr)
                    event_outq.put(("state:finished", exc_type, {"stderr": result.stderr}, None))
                else:
                    event_outq.put(("state:finished", None, None, None))
                    
            except sp.TimeoutExpired as e:
                result_outq.put(f"TimeoutError: Execution exceeded {self.timeout} seconds")
                if e.stdout:
                    result_outq.put(e.stdout if isinstance(e.stdout, str) else e.stdout.decode())
                if e.stderr:
                    result_outq.put(e.stderr if isinstance(e.stderr, str) else e.stderr.decode())
                event_outq.put(("state:finished", "TimeoutError", {"timeout": self.timeout}, None))
                
            except Exception as e:
                error_msg = f"Execution error: {type(e).__name__}: {str(e)}"
                result_outq.put(error_msg)
                event_outq.put(("state:finished", type(e).__name__, {"error": str(e)}, None))

            # put EOF marker to indicate that we're done
            result_outq.put("<|EOF|>")

    def _parse_exception_type(self, stderr: str) -> str:
        """Parse exception type from stderr text."""
        if not stderr:
            return "RuntimeError"
        
        # Look for common Python exception patterns
        import re
        lines = stderr.strip().split('\n')
        for line in reversed(lines):
            # Match patterns like "ValueError: message" or "ModuleNotFoundError: No module named 'x'"
            match = re.match(r'^(\w+Error|\w+Exception|KeyboardInterrupt):', line)
            if match:
                return match.group(1)
        
        # Default to RuntimeError if pattern not found
        return "RuntimeError"

    def create_process(self) -> None:
        # Use 'spawn' context to avoid fork deadlock in multi-threaded environment
        # This is critical when running after OpenHands or other threaded operations
        ctx = get_context('spawn')
        
        # we use three queues to communicate with the child process:
        # - code_inq: send code to child to execute
        # - result_outq: receive stdout/stderr from child
        # - event_outq: receive events from child (e.g. state:ready, state:finished)
        self.code_inq = ctx.Queue()
        self.result_outq = ctx.Queue()
        self.event_outq = ctx.Queue()
        
        # Package all necessary config into a picklable dict
        config = {
            'working_dir': str(self.working_dir),
            'timeout': self.timeout,
            'python_path': self.python_path,
            'env_vars': self.env_vars,
            'agent_file_name': self.agent_file_name,
        }
        
        self.process = ctx.Process(
            target=_run_session_worker,  # Use top-level function for spawn compatibility
            args=(config, self.code_inq, self.result_outq, self.event_outq),
        )
        self.process.start()
        logger.info(f"🚀 Created subprocess with spawn context (PID: {self.process.pid})")

    def _drain_queues(self):
        """Quickly drain all in-flight messages to prevent blocking."""
        drained_count = 0
        
        # Drain result queue
        while not self.result_outq.empty():
            try:
                self.result_outq.get_nowait()
                drained_count += 1
            except Exception:
                break

        # Drain event queue
        while not self.event_outq.empty():
            try:
                self.event_outq.get_nowait()
                drained_count += 1
            except Exception:
                break

        # Drain code queue
        while not self.code_inq.empty():
            try:
                self.code_inq.get_nowait()
                drained_count += 1
            except Exception:
                break
        
        if drained_count > 0:
            logger.debug(f"Drained {drained_count} queue messages")

    def cleanup_session(self):
        if self.process is None:
            return
        
        logger.info(f"Cleaning up process PID={self.process.pid}")
        
        # Step 0: Kill experiment process group (nnUNet + DataLoader workers + all descendants)
        if self._experiment_pgid:
            try:
                os.killpg(self._experiment_pgid, signal.SIGKILL)
                logger.info(f"✅ Killed experiment process group PGID={self._experiment_pgid}")
            except (ProcessLookupError, OSError):
                logger.debug(f"Experiment process group PGID={self._experiment_pgid} already dead")
            self._experiment_pgid = None
        
        # Step 1: Graceful termination
        try:
            self.process.terminate()
            self._drain_queues()
            self.process.join(timeout=3)
        except Exception as e:
            logger.warning(f"Graceful termination failed: {e}")
        
        # Step 2: Force kill if still alive
        if self.process.exitcode is None:
            logger.warning("Force killing process...")
            try:
                self.process.kill()
                self._drain_queues()
                self.process.join(timeout=2)
            except Exception as e:
                logger.error(f"Force kill failed: {e}")
        
        # Step 3: Force cleanup process tree
        self._force_cleanup()
        
        # Step 4: Clean up resources
        try:
            self.process.close()
        except Exception:
            pass
        self.process = None

    def _force_cleanup(self):
        """Force cleanup of entire process tree using psutil with timeout protection"""
        if self.process is None:
            return
        
        import threading
        import time
        
        cleanup_finished = threading.Event()
        cleanup_exception = []
        
        def _cleanup_worker():
            """Worker function to perform cleanup with timeout protection"""
            try:
                import psutil
                
                # Get the main process with timeout protection
                try:
                    parent = psutil.Process(self.process.pid)
                except psutil.NoSuchProcess:
                    logger.info("Main process already terminated")
                    return
                
                # Get all children (recursive) with timeout protection  
                try:
                    children = parent.children(recursive=True)
                    logger.info(f"Found {len(children)} child processes to cleanup")
                except Exception as e:
                    logger.warning(f"Failed to get children: {e}")
                    children = []
                
                # Add parent to the list
                all_processes = children + [parent]
                
                # Step 1: Send SIGTERM to all
                terminated_count = 0
                for proc in all_processes:
                    try:
                        proc.terminate()
                        terminated_count += 1
                        logger.debug(f"Sent SIGTERM to PID {proc.pid}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                logger.info(f"Sent SIGTERM to {terminated_count} processes")
                
                # Step 2: Wait briefly for graceful shutdown (with reduced timeout)
                try:
                    gone, alive = psutil.wait_procs(all_processes, timeout=2)
                    logger.info(f"Terminated {len(gone)} processes, {len(alive)} still alive")
                except Exception as e:
                    logger.warning(f"wait_procs failed: {e}")
                    alive = all_processes  # Assume all still alive
                
                # Step 3: Force kill survivors (with individual timeouts)
                killed_count = 0
                for proc in alive:
                    try:
                        proc.kill()
                        killed_count += 1
                        logger.debug(f"Force killed PID {proc.pid}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                if killed_count > 0:
                    logger.warning(f"Force killed {killed_count} processes")
                
            except ImportError:
                logger.warning("psutil not available in cleanup worker")
                cleanup_exception.append("psutil not available")
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
                cleanup_exception.append(e)
            finally:
                cleanup_finished.set()
        
        # Start cleanup in a separate thread with timeout
        cleanup_thread = threading.Thread(target=_cleanup_worker, daemon=True)
        cleanup_thread.start()
        
        # Wait for cleanup with timeout (max 10 seconds)
        if cleanup_finished.wait(timeout=10):
            logger.info("Cleanup completed within timeout")
        else:
            logger.error("Cleanup timed out after 10 seconds, continuing anyway")
        
        # Log any exceptions that occurred during cleanup
        if cleanup_exception:
            logger.error(f"Cleanup exceptions: {cleanup_exception}")

    def _fallback_cleanup(self):
        """Fallback cleanup when psutil is not available - with timeout protection"""
        if self.process is None:
            return
        
        import threading
        
        cleanup_finished = threading.Event()
        
        def _fallback_worker():
            """Worker function for fallback cleanup"""
            try:
                # Try to kill the main process with SIGKILL
                os.kill(self.process.pid, signal.SIGKILL)
                logger.info(f"Sent SIGKILL to main process {self.process.pid}")
                
                # Try to kill any potential multiprocessing children
                # This is a best-effort approach without psutil
                import subprocess
                try:
                    # Find children of our process with timeout
                    result = subprocess.run(
                        ['pgrep', '-P', str(self.process.pid)], 
                        capture_output=True, text=True, timeout=3
                    )
                    if result.returncode == 0:
                        child_pids = result.stdout.strip().split('\n')
                        killed_count = 0
                        for pid in child_pids:
                            if pid.strip():
                                try:
                                    os.kill(int(pid), signal.SIGKILL)
                                    killed_count += 1
                                    logger.debug(f"Killed child process {pid}")
                                except Exception:
                                    pass
                        if killed_count > 0:
                            logger.info(f"Killed {killed_count} child processes")
                except subprocess.TimeoutExpired:
                    logger.warning("pgrep command timed out")
                except Exception as e:
                    logger.warning(f"Failed to find child processes: {e}")
                    
            except (ProcessLookupError, OSError):
                # Process already dead
                logger.debug("Process already terminated")
            except Exception as e:
                logger.error(f"Fallback cleanup worker failed: {e}")
            finally:
                cleanup_finished.set()
        
        # Start fallback cleanup in thread with timeout
        cleanup_thread = threading.Thread(target=_fallback_worker, daemon=True)
        cleanup_thread.start()
        
        # Wait for cleanup with timeout (max 5 seconds)
        if cleanup_finished.wait(timeout=5):
            logger.info("Fallback cleanup completed")
        else:
            logger.error("Fallback cleanup timed out after 5 seconds")

    def run_with_escape_hatch(self, code: str, reset_session=True) -> ExecutionResult:
        """
        Execute code with an escape hatch - guaranteed to return within timeout + 45 seconds.
        
        This method wraps the original run() with a daemon thread and absolute timeout.
        If the execution gets stuck (even during cleanup), it will abandon the process
        and return an error result to allow the experiment to continue.
        
        Parameters:
            code (str): Python code to execute.
            reset_session (bool, optional): Whether to reset the interpreter session. Defaults to True.
            
        Returns:
            ExecutionResult: Object containing the output and metadata, or an abandoned error result.
        """
        import threading
        
        result_container = []
        exception_container = []
        
        def run_worker():
            """Worker function to execute run() in a separate thread"""
            try:
                result = self.run(code, reset_session)
                result_container.append(result)
            except Exception as e:
                exception_container.append(e)
                logger.error(f"Exception in run worker: {e}")
        
        # Calculate absolute maximum wait time
        max_total_time = self.timeout + 45 if self.timeout else 3645
        
        logger.info(f"🚀 Starting execution with escape hatch (max: {max_total_time}s)")
        
        # Start execution in daemon thread (will be killed if main thread exits)
        worker_thread = threading.Thread(target=run_worker, daemon=True)
        worker_thread.start()
        
        # Wait for completion with absolute timeout
        worker_thread.join(timeout=max_total_time)
        
        if worker_thread.is_alive():
            # Thread still running = stuck somewhere, abandon it!
            logger.error(f"🚨 ESCAPE HATCH TRIGGERED: Execution stuck after {max_total_time}s")
            logger.error(f"🚨 Abandoning process to prevent infinite hang")
            logger.warning(f"⚠️ Process PID {self.process.pid if self.process else 'unknown'} may be leaked")
            
            # Force cleanup attempt (non-blocking)
            try:
                if self.process:
                    logger.info("🗑️ Attempting emergency cleanup...")
                    self._emergency_cleanup()
            except Exception as e:
                logger.error(f"Emergency cleanup failed: {e}")
            
            # Return timeout error result (escape hatch triggered)
            return ExecutionResult(
                term_out=[
                    f"TimeoutError: Process stuck after {max_total_time} seconds (escape hatch triggered)",
                    f"This usually indicates a severe hang in code execution or cleanup.",
                    f"The process may have been leaked - check system resources.",
                    f"Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
                ],
                exec_time=max_total_time,
                exc_type="TimeoutError",
                exc_info={"reason": "Escape hatch triggered - execution stuck beyond timeout"},
                exc_stack=[]
            )
        
        # Thread completed within timeout
        if result_container:
            logger.info(f"✅ Execution completed successfully")
            return result_container[0]
        elif exception_container:
            logger.error(f"❌ Execution raised exception: {exception_container[0]}")
            # Re-raise the exception to be handled by caller
            raise exception_container[0]
        else:
            # Should not happen, but handle gracefully
            logger.error("❌ Execution completed but no result received")
            return ExecutionResult(
                term_out=[
                    "TimeoutError: Execution completed but no result received",
                    f"Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
                ],
                exec_time=max_total_time,
                exc_type="TimeoutError",
                exc_info={"reason": "No result received after execution"},
                exc_stack=[]
            )

    def _emergency_cleanup(self):
        """Emergency cleanup attempt - non-blocking, best effort"""
        # Kill experiment process group first
        if self._experiment_pgid:
            try:
                os.killpg(self._experiment_pgid, signal.SIGKILL)
                logger.info(f"Emergency: killed experiment process group PGID={self._experiment_pgid}")
            except (ProcessLookupError, OSError):
                pass
            self._experiment_pgid = None
        try:
            if self.process and self.process.pid:
                os.kill(self.process.pid, signal.SIGKILL)
                logger.info(f"Emergency: sent SIGKILL to spawn worker PID={self.process.pid}")
        except Exception as e:
            logger.debug(f"Emergency cleanup failed: {e}")

    def _read_output_with_timeout(self, max_wait_time: int = 10) -> List[str]:
        """
        Read output from result queue with timeout protection.
        
        This prevents infinite hangs when the child process is killed before
        sending the EOF marker or when the queue is in an inconsistent state.
        
        Parameters:
            max_wait_time (int): Maximum time to wait for output in seconds
            
        Returns:
            List[str]: Output lines (without EOF marker)
        """
        output: List[str] = []
        start_time = time.time()
        messages_read = 0
        eof_received = False
        
        logger.debug(f"📖 Starting output read (max: {max_wait_time}s)")
        
        try:
            while True:
                elapsed = time.time() - start_time
                
                # Check total timeout
                if elapsed > max_wait_time:
                    logger.warning(f"⏰ Output read timed out after {elapsed:.1f}s")
                    break
                
                # Check if we have EOF marker
                if output and output[-1] == "<|EOF|>":
                    eof_received = True
                    output.pop()  # Remove EOF marker
                    logger.debug(f"✅ Received EOF marker after reading {messages_read} messages")
                    break
                
                # Try to read next message with timeout
                try:
                    remaining_time = max(1, max_wait_time - elapsed)
                    msg = self.result_outq.get(timeout=min(1, remaining_time))
                    output.append(msg)
                    messages_read += 1
                except queue.Empty:
                    # Queue is empty, check if we should continue waiting
                    if output and elapsed > 3:
                        # We have some output and waited a while, assume done
                        logger.debug(f"📭 Queue empty after {elapsed:.1f}s, assuming complete")
                        break
                    # Otherwise continue waiting (queue might still be filling)
                    continue
                
        except Exception as e:
            logger.error(f"❌ Error reading output: {e}")
        
        # Log final status
        if not eof_received:
            logger.warning(f"⚠️ EOF marker not received (read {messages_read} messages in {time.time()-start_time:.1f}s)")
        
        logger.debug(f"📊 Output read complete: {messages_read} messages, EOF: {eof_received}")
        
        return output

    def run(self, code: str, reset_session=True) -> ExecutionResult:
        """
        Execute the provided Python command in a separate process and return its output.

        Parameters:
            code (str): Python code to execute.
            reset_session (bool, optional): Whether to reset the interpreter session before executing the code. Defaults to True.

        Returns:
            ExecutionResult: Object containing the output and metadata of the code execution.

        """

        logger.debug(f"REPL is executing code (reset_session={reset_session})")

        if reset_session:
            if self.process is not None:
                # terminate and clean up previous process
                self.cleanup_session()
            self.create_process()
        else:
            # reset_session needs to be True on first exec
            assert self.process is not None

        assert self.process.is_alive()

        # First, wait for worker to be ready (spawn initialization can take time)
        try:
            startup_state = self.event_outq.get(timeout=30)  # Spawn can take longer to start
            if startup_state[0] == "state:error":
                # Worker startup failed
                error_info = startup_state[2] if len(startup_state) > 2 else {}
                error_msg = error_info.get("error", "Unknown worker startup error")
                logger.critical(f"Worker startup failed: {error_msg}")
                raise RuntimeError(f"Worker startup failed: {error_msg}")
            elif startup_state[0] != "state:worker_started":
                logger.warning(f"Unexpected startup state: {startup_state}")
        except queue.Empty:
            msg = "REPL worker failed to start (spawn timeout after 30s)"
            logger.critical(msg)
            # Try to get any error messages from result queue
            while not self.result_outq.empty():
                logger.error(f"REPL output queue dump: {self.result_outq.get()}")
            raise RuntimeError(msg) from None
        
        logger.debug("Worker started successfully, sending code...")
        self.code_inq.put(code)

        # wait for child to actually start execution (we don't want interrupt child setup)
        try:
            state = self.event_outq.get(timeout=10)
        except queue.Empty:
            msg = "REPL child process failed to start execution"
            logger.critical(msg)
            while not self.result_outq.empty():
                logger.error(f"REPL output queue dump: {self.result_outq.get()}")
            raise RuntimeError(msg) from None
        assert state[0] == "state:ready", state
        # Extract experiment subprocess PID for process-group cleanup
        self._experiment_pgid = state[1] if len(state) > 1 else None
        if self._experiment_pgid:
            logger.info(f"📌 Experiment subprocess PID/PGID: {self._experiment_pgid}")
        start_time = time.time()

        # this flag indicates that the child ahs exceeded the time limit and an interrupt was sent
        # if the child process dies without this flag being set, it's an unexpected termination
        child_in_overtime = False

        while True:
            try:
                # check if the child is done
                state = self.event_outq.get(timeout=1)  # wait for state:finished
                assert state[0] == "state:finished", state
                exec_time = time.time() - start_time
                break
            except queue.Empty:
                # we haven't heard back from the child -> check if it's still alive (assuming overtime interrupt wasn't sent yet)
                if not child_in_overtime and not self.process.is_alive():
                    msg = "REPL child process died unexpectedly"
                    logger.critical(msg)
                    while not self.result_outq.empty():
                        logger.error(
                            f"REPL output queue dump: {self.result_outq.get()}"
                        )
                    raise RuntimeError(msg) from None

                # child is alive and still executing -> check if we should sigint..
                if self.timeout is None:
                    continue
                running_time = time.time() - start_time
                if running_time > self.timeout:
                    assert reset_session, "Timeout occurred in interactive session"
                    
                    if not child_in_overtime:
                        # First timeout detection
                        logger.warning(f"⏰ Timeout exceeded: {running_time:.1f}s > {self.timeout}s")
                        logger.info(f"🔧 Stage 1/3: Sending termination signal...")
                        
                        # Step 1a: Kill experiment process group first (nnUNet + DataLoader workers)
                        if self._experiment_pgid:
                            try:
                                os.killpg(self._experiment_pgid, signal.SIGTERM)
                                logger.info(f"✅ Sent SIGTERM to experiment process group PGID={self._experiment_pgid}")
                            except (ProcessLookupError, OSError) as e:
                                logger.warning(f"⚠️ Could not SIGTERM experiment process group: {e}")
                        
                        # Step 1b: Also signal the spawn worker
                        try:
                            os.kill(self.process.pid, signal.SIGTERM)
                            logger.info(f"✅ Sent SIGTERM to spawn worker PID={self.process.pid}")
                        except (ProcessLookupError, OSError) as e:
                            logger.warning(f"⚠️ Could not send SIGTERM to spawn worker: {e}")
                        
                        child_in_overtime = True
                    else:
                        # Already in overtime, just log occasionally
                        overtime = running_time - self.timeout
                        if int(overtime) % 5 == 0:  # Log every 5 seconds
                            logger.debug(f"⏳ Still in overtime: +{overtime:.0f}s")
                    
                    # Step 2: Force cleanup after 15 second grace period
                    if running_time > self.timeout + 15:
                        logger.error(f"🚨 Grace period expired ({running_time - self.timeout:.1f}s overtime)")
                        logger.info(f"🔧 Stage 2/3: Force killing process tree...")
                        
                        # Force SIGKILL on experiment process group first
                        if self._experiment_pgid:
                            try:
                                os.killpg(self._experiment_pgid, signal.SIGKILL)
                                logger.info(f"✅ Sent SIGKILL to experiment process group PGID={self._experiment_pgid}")
                            except (ProcessLookupError, OSError) as e:
                                logger.debug(f"Experiment process group already dead: {e}")
                        
                        cleanup_start = time.time()
                        cleanup_success = False
                        
                        # Try psutil-based cleanup first
                        if HAS_PSUTIL:
                            try:
                                logger.info("🔍 Using enhanced cleanup (psutil)")
                                self._force_cleanup()
                                cleanup_success = True
                                logger.info(f"✅ Enhanced cleanup completed in {time.time()-cleanup_start:.1f}s")
                            except Exception as e:
                                logger.error(f"❌ Enhanced cleanup failed: {e}")
                        else:
                            logger.info("ℹ️ psutil not available, using basic cleanup")
                        
                        # Fallback to basic cleanup if psutil failed or not available
                        if not cleanup_success:
                            try:
                                logger.info("🔧 Attempting fallback cleanup...")
                                self._fallback_cleanup()
                                logger.info(f"✅ Fallback cleanup completed in {time.time()-cleanup_start:.1f}s")
                            except Exception as e:
                                logger.error(f"❌ Fallback cleanup failed: {e}")
                        
                        # Step 3: Final cleanup and exit
                        logger.info(f"🔧 Stage 3/3: Final cleanup and exit...")
                        try:
                            self.cleanup_session()
                            logger.info("✅ Final cleanup completed")
                        except Exception as e:
                            logger.error(f"❌ Final cleanup failed: {e}")
                        
                        total_cleanup_time = time.time() - cleanup_start
                        logger.info(f"📊 Total cleanup time: {total_cleanup_time:.1f}s")
                        logger.info(f"🔄 Returning TimeoutError result, experiment will continue")
                        
                        state = (None, "TimeoutError", {}, [])
                        exec_time = self.timeout
                        break

        # Read output with timeout protection to prevent infinite hang
        output = self._read_output_with_timeout(max_wait_time=10)

        e_cls_name, exc_info, exc_stack = state[1:]

        if e_cls_name == "TimeoutError":
            output.append(
                f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
            )
        else:
            output.append(
                f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(self.timeout)})."
            )
        return ExecutionResult(output, exec_time, e_cls_name, exc_info, exc_stack)
