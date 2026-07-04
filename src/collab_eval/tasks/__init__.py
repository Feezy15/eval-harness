"""Task registry: maps config `tasks[].name` strings to Task classes.

Adding a task = one module implementing Task + one line here.
"""

from collab_eval.tasks.base import Task
from collab_eval.tasks.toy import ToyTask

TASK_REGISTRY: dict[str, type[Task]] = {
    ToyTask.name: ToyTask,
}

__all__ = ["TASK_REGISTRY", "Task"]