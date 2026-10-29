"""NoRA-init numerical, autograd, resume and dynamic inference acceptance."""
import runpy
import sys
from pathlib import Path
sys.argv += ['--method', 'nora']
runpy.run_path(str(Path(__file__).with_name('check_miss.py')), run_name='__main__')
