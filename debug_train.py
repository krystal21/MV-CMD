# debugProxy.py
import os, sys, runpy

## 1. cd WORKDIR
# os.chdir('WORKDIR')

args = "python temp.py"
# args = "python train_linear.py --config config/config_viewcon.yml   --gpu_id 0 "
# args = "python -m mymodule.test 4 5"

args = args.split()
if args[0] == "python":
    """pop up the first in the args"""
    args.pop(0)
if args[0] == "-m":
    """pop up the first in the args"""
    args.pop(0)
    fun = runpy.run_module
else:
    fun = runpy.run_path
sys.argv.extend(args[1:])
fun(args[0], run_name="__main__")
