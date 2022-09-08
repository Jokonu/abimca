# Contributing

Thank you for taking the time and interest to contribute to this project.
You are welcome to
- create a bug report
- submit a pull request
- suggest improvement or feature


## Development steps

If you plan on contribute code here are some commands to help keep the code clean.

### Formatting

Formatting is done with black and isort
.. code:: bash

    $ black abimca
    $ isort . --profile black

### Linting

For this project we use pylint and flake8

    $ pylint abimca -d "C0301,C0103,E0401" --fail-under=8
    $ flake8 abimca --max-line-length=120 --ignore=E501,W503

### Type checking

    $ mypy abimca


### Testing
TODO!

### Version bump and release

using bumpver:

    $ bumpver update --patch
    # or
    $ bumpver update --minor
    # or
    $ bumpver update --major
    # for a testrun:
    $ bumpver update --patch --dry -n

and publish to testpypi:
register repository first if not done yet
```bash
poetry config repositories.test-pypi https://test.pypi.org/legacy/
```
for publishing to testPyPi then use:
```bash
poetry publish --repository test-pypi --username __token__ --password test-pypi-token-here --build
```

test the installation with pip
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ abimca
```
