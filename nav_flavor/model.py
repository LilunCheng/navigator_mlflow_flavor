import pickle
from pathlib import Path
import model_navigator as nav


class NavModel:
    def __init__(self, pkg_desc):
        self.pkg_desc = pkg_desc

    def predict(self, inputs):
        #Implement in the future
        return None

    def save(self, path):
        nav.save(self.pkg_desc, Path(path))

    @classmethod
    def load(cls, path):
        return nav.load(path, retest_conversions=False)
