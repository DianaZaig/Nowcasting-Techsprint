#!python.exe -m pip install --upgrade pip
import subprocess
import importlib


def install_package(package_name):
    try:
        importlib.import_module(package_name)
    except ImportError:
        subprocess.check_call(["pip", "install", package_name])

def install_missing_packages():
    install_package("matplotlib")
    install_package("numpy")
    install_package("pandas")
    install_package("scikit-learn")
    install_package("pickle")
    install_package("tensorflow")
    install_package("imbalanced-learn")
    install_package("seaborn")
    install_package("datetime")
    install_package("pickle")









