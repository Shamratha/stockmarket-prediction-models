"""Run the entire Stock Market suite: forecasting zoo, trading agents,
simulations. Charts land in output/, tables in output/*.csv.

Usage: python scripts/run_all.py [TICKER]
"""

import runpy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

for script in ('run_forecasting.py', 'run_agents.py', 'run_simulations.py'):
    print(f'\n=== {script} ===')
    sys.argv = [script] + sys.argv[1:]
    runpy.run_path(os.path.join(HERE, script), run_name='__main__')
