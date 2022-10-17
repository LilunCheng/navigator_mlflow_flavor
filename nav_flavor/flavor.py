from mlflow.models import Model
from nav_flavor.model import NavModel
from mlflow.models.model import MLMODEL_FILE_NAME
from pathlib import Path
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
import nav_flavor


FLAVOR_NAME = "nav"


# options commented out are not necessary
def save_model(
    nav_model: NavModel,
    path,
    # conda_env=None,
    mlflow_model=None,
    # code_paths=None,
    # signature: ModelSignature = None,
    # input_example: ModelInputExample = None,
    # requirements_file=None,
    # extra_files=None,
    # pip_requirements=None,
    # extra_pip_requirements=None,
    navigator_version="0.3.3"
):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)

    mlflow_mlmodel_file_path = path / MLMODEL_FILE_NAME

    model_subpath = path / "model.nav"

    if mlflow_model is None:
        mlflow_model = Model()

    mlflow_model.add_flavor(FLAVOR_NAME, navigator_version=navigator_version)
    mlflow_model.save(mlflow_mlmodel_file_path)
    nav_model.save(model_subpath)


def load_model(model_uri, dst_path=None):
    local_model_path = _download_artifact_from_uri(
        artifact_uri=model_uri, output_path=dst_path
    )
    model_subpath = Path(local_model_path) / "model.nav"
    return NavModel.load(model_subpath)


def log_model(
    model: NavModel,
    artifact_path,
    registered_model_name=None,
    # conda_env=None,
    # code_paths=None,
    # registered_model_name=None,
    # signature: ModelSignature = None,
    # input_example: ModelInputExample = None,
    # pip_requirements=None,
    # extra_pip_requirements=None,
    **kwargs,
):
    return Model.log(
        artifact_path=str(artifact_path),  # must be string, numbers etc
        flavor=nav_flavor.flavor,  # points to this module itself
        registered_model_name=registered_model_name,
        nav_model=model,
        # conda_env=conda_env,
        # code_paths=code_paths,
        # signature=signature,
        # input_example=input_example,
        # pip_requirements=pip_requirements,
        # extra_pip_requirements=extra_pip_requirements,
        **kwargs,
    )
