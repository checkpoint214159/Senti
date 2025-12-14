import glob
import importlib
import os

# lazily import all .py files 
package_dir = os.path.dirname(__file__)

module_files = glob.glob(os.path.join(package_dir, "*.py"))
module_names = [
    os.path.splitext(os.path.basename(f))[0]
    for f in module_files
    if os.path.basename(f) != "__init__.py"
]

for module_name in module_names:
    # relative import syntax: .<module_name>
    importlib.import_module(f".{module_name}", package=__name__)

