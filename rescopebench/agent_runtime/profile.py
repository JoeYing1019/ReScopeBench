"""Pinned evaluation defaults; run plans record model-specific choices."""

MODEL = 'gpt-5.5-2026-04-24'

SEED = 20260922

MAX_TOKENS = 65536

PROFILE = 'model:\n  streaming: false\nagent:\n  api_max_retries: 0\n  coding_context: "off"\n  tool_use_enforcement: false\nmemory:\n  memory_enabled: true\n  memory_char_limit: 16000\n  user_char_limit: 1375\nskills:\n  external_dirs: []\n  trusted_project_dirs: []\ntools:\n  tool_search:\n    enabled: false\nauxiliary:\n  title_generation:\n    enabled: false\n  background_review:\n    enabled: true\n    extra_tools:\n      - episode_list\n      - episode_read\n      - record_search\n      - record_read\ntelemetry:\n  shared_metrics:\n    enabled: false\n'

ADOPTION_ALLOWLIST = ['*']

TASK_PREFIX = 'Below is the task for this episode:\n\n'

SAVE_PROMPT = 'After completing or stopping this task, review whether the episode produced\nexperience that may help with future tasks in the same environment. If so,\nupdate persistent experience before finishing.\n\nUse memory for concrete environment-specific facts, and use skills for reusable\ndecision rules or procedures. Use your own judgment about what is worth\nretaining and how much detail is useful.\n\nWhen new evidence conflicts with prior experience, reconcile the conflict and\nrevise what you would rely on going forward. If nothing useful was learned, do\nnot force an update.'

REFLECT_PREFIX = 'Briefly reflect on consequential choices, observed failures, and observed outcomes. '

MAX_ITERATIONS = 100

PARALLEL_TOOL_CALLS = False

SYSTEM = 'Operate only through the declared tools and complete the current task in the local environment. Treat task-declared events and tool observations as evidence; do not assume unobserved structure, record locations, or transition behavior. Prior task records may be incomplete or wrong; verify reused experience when it matters.'

PHASES = {'native_review', 'task'}

PROTOCOL = 'robot-state-action-native-eval/v2'
