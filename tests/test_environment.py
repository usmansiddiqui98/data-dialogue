import os
import re
import sys

REQUIRED_PYTHON = "python3"


def test_python_version():
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError("Unrecognized python interpreter: {}".format(REQUIRED_PYTHON))

    assert system_major == required_major, "This project requires Python {}. Found: Python {}".format(
        required_major, sys.version
    )


# List of package names to exclude
exclude_packages = [
    "re",
    "os",
    "sys",
    "math",
    "random",
    "logging",
    "string",
]


def test_development_environment():
    # Get the path of the requirements.txt file
    req_file_path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")

    # Check that all packages in src are listed in requirements.txt
    with open(req_file_path) as f:
        requirements = f.read().splitlines()
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                with open(filepath) as f:
                    content = f.read()
                for line in content.splitlines():
                    if "import" in line:
                        package_match = re.search(r"^\s*import\s+(\w+)", line)
                        if package_match:
                            package = package_match.group(1)
                            if (
                                "src" not in package
                                and package not in sys.builtin_module_names
                                and package not in exclude_packages
                            ):
                                assert (
                                    package in requirements
                                ), f"Package {package} used in {filepath} is not listed in {req_file_path}"
