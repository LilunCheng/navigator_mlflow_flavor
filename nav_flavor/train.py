"""Execute the script by providing an offset. For example:

python mlflow_flavor_example/train.py --offset 0.3
"""

#pip install --extra-index-url https://pypi.ngc.nvidia.com git+https://github.com/triton-inference-server/model_navigator.git@v0.3.3
#export MLFLOW_TRACKING_URI="http://localhost:5000"
import argparse
from nav_flavor.model import NavModel
from nav_flavor.flavor import log_model
import mlflow
import model_navigator as nav

def main():

    pkg_dsc = nav.load('/home/lilun/work/projects/nvidia_tensorRT/triton_torch/example_pyt.nav', retest_conversions=False)
    with mlflow.start_run() as run:
        artifact_directory = "nav_models"
        model = NavModel(pkg_dsc)
        log_model(model, artifact_path=artifact_directory, registered_model_name='simple_torch_example_model')

if __name__ == "__main__":
    main()
