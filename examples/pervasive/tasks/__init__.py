import importlib
import os
from fairseq.tasks import TASK_REGISTRY

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        task_name = file[:file.find('.py')]

        # Avoid duplicate import if task is already registered
        if task_name not in TASK_REGISTRY:
            importlib.import_module('examples.pervasive.tasks.' + task_name)
