"""
.. _first:

How to Implement and Register Your Algorithm with OptunaHub
===========================================================

OptunaHub is an Optuna package registry, which is a user platform to share their optimization algorithms.
This recipe shows how to implement and register your own sampling algorithm with OptunaHub.

How to Implement Your Own Algorithm
-----------------------------------

Usually, Optuna provides `BaseSampler` class to implement your own sampler.
However, it is a bit complicated to implement a sampler from scratch.
Instead, in OptunaHub, you can use `samplers/simple/SimpleSampler` class, which is a sampler template that can be easily extended.

You need to install `optuna` to implement your own sampler, and `optunahub` to use the template `SimpleSampler`.

.. code-block:: bash

    $ pip install optuna optunahub

"""

###################################################################################################
# First of all, import `optuna`, `optunahub`, and other required modules.
from __future__ import annotations

import os
from typing import Any

from github import Auth
import numpy as np
import optuna
import optunahub


###################################################################################################
# Next, define your own sampler class by inheriting `SimpleSampler` class.
# In this example, we implement a sampler that returns a random value.
# `SimpleSampler` class can be loaded using `optunahub.load_module` function.
# To download the registered sampler, we need to set `force_reload` to `False`.
# `force_reload=True` argument forces downloading the sampler from the registry.
# If we set `force_reload` to `False`, we use the cached data in our local storage if available.

SimpleSampler = optunahub.load_module(
    "samplers/simple",
    auth=Auth.Token(os.environ["SECRET_GITHUB_TOKEN"]),
).SimpleSampler


class MySampler(SimpleSampler):  # type: ignore
    # `search_space` argument is necessary for the concrete implementation of `SimpleSampler` class.
    def __init__(self, search_space: dict[str, optuna.distributions.BaseDistribution]) -> None:
        super().__init__(search_space)
        self._rng = np.random.RandomState()

    # You need to implement `sample_relative` method.
    # This method returns a dictionary of hyperparameters.
    # The keys of the dictionary are the names of the hyperparameters, which must be the same as the keys of the `search_space` argument.
    # The values of the dictionary are the values of the hyperparameters.
    # In this example, `sample_relative` method returns a dictionary of randomly sampled hyperparameters.
    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, Any]:
        # `search_space` argument must be identical to `search_space` argument input to `__init__` method.
        # This method is automatically invoked by Optuna and `SimpleSampler`.

        params = {}
        for n, d in search_space.items():
            if isinstance(d, optuna.distributions.FloatDistribution):
                params[n] = self._rng.uniform(d.low, d.high)
            elif isinstance(d, optuna.distributions.IntDistribution):
                params[n] = self._rng.randint(d.low, d.high)
            elif isinstance(d, optuna.distributions.CategoricalDistribution):
                params[n] = d.choices[0]
            else:
                raise NotImplementedError
        return params


###################################################################################################
# In this example, the objective function is a simple quadratic function.


def objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    return x**2


###################################################################################################
# This sampler can be used in the same way as other Optuna samplers.
# In the following example, we create a study and optimize it using `MySampler` class.
sampler = MySampler({"x": optuna.distributions.FloatDistribution(-10, 10)})
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=100)

###################################################################################################
# The best parameter can be fetched as follows.

best_params = study.best_params
found_x = best_params["x"]
print(f"Found x: {found_x}, (x - 2)^2: {(found_x - 2) ** 2}")

###################################################################################################
# We can see that ``x`` value found by Optuna is close to the optimal value ``2``.

###################################################################################################
# How to Register Your Implemented Algorithm with OptunaHub
# ---------------------------------------------------------
#
# After implementing your own algorithm, you can register it with OptunaHub.
# You need to create a pull request to the `optunahub-registry <https://github.com/optuna/optunahub-registry>`_ repository.
#
# The following is an example of the directory structure of the pull request.
#
# | package
# | └── samplers
# |     └── YOUR_ALGORITHM_NAME
# |         ├── README.md
# |         ├── __init__.py
# |         ├── LICENSE
# |         ├── images
# |         │   ├──  thumbnail.png
# |         │   └──  screenshot.png
# |         ├── requirements.txt
# |         └── YOUR_ALGORITHM_NAME.py
#
# An implemented sampler should be put in the `samplers` directory.
# In the `samplers` directory, you should create a directory named after your algorithm.
# The created directory should include the following files:
#
# - `README.md`: A description of your algorithm. This file is used to create an `web page of OptunaHub <TBD>`_. Let me explain the format of the `README.md` file later.
# - `__init__.py`: An initialization file. This file must import your implemented sampler from `YOUR_ALGORITHM_NAME.py`.
# - `LICENSE`: A license file. This file must contain the license of your algorithm. It should be the MIT license in the alpha version.
# - `images`: This is optional. A directory that contains images. The images in this directory will be used the `web page of OptunaHub <TBD>`_. `thumbnail.png` will be used as a thumbnail in the web page. Note that `README.md` can also refer to image files, e.g. `images/screenshot.png`,  in this directory.
# - `requirements.txt`: This is optional. A file that contains the additional dependencies of your algorithm. If there are no additional dependencies, you do not need to create this file.
# - `YOUR_ALGORITHM_NAME.py`: Your implemented sampler.
#
# `README.md` must contain the following sections:
#
# - A header section written in the following format:
#
#   .. code-block:: markdown
#
#       ---
#       author: 'Optuna team'
#       title: 'Demo Sampler'
#       description: 'Demo Sampler of OptunaHub'
#       tags: ['sampler']
#       optuna_versions: ['3.6.1']
#       license: 'MIT License'
#       ---
#
#   - `author`: The author of the package. It can be your name or your organization name.
#   - `title`: The package title. It should not be a class/function name but a human-readable name. For example, `Demo Sampler` is a good title, but `DemoSampler` is not.
#   - `description`: A brief description of the package. It should be a one-sentence summary of the package.
#   - `tags`: The package tags. It should be a list of strings. The tags must include `sampler` or `visualization` depending on the type of the package. You can add other tags as needed. For example, "['sampler', 'LLM']".
#   - `optuna_versions`: A list of Optuna versions that the package supports. It should be a list of strings. For example, "['3.5.0', '3.6.1']".
#   - `license`: The license of the package. It should be a string. For example, `'MIT License'`. The license must be `MIT` in the alpha version.
#
# - `Class or Function Names` section that describes the classes or functions provided by the package. If you provide multiple classes or functions, you should list them in this section. Note that the section must be a markdown list. If you provide only one class or function, you can simply write the class or function name. Note that the documentation of the classes or functions must be written in their docstrings. If you want to refer to the documentation, please leave the source code link, or write them in the following `Others` section. For example:
#
#   .. code-block:: markdown
#
#       - `DemoSampler1`
#       - `DemoSampler2`
#       - `demo_function1`
#
# - An `Installation` section that describes how to install the additional dependencies if required. For example:
#
#   .. code-block:: markdown
#
#       $ pip install -r requirements.txt
#
# - An `Example` section that describes how to use the package. It should be a python code block. It should be a few lines of code snippets that show how to use the package. If you want to provide a full example, please create a separete file like `example.py` and refer to it. For example:
#
#   .. code-block:: markdown
#
#       ```python
#       sampler = DemoSampler()
#       study = optuna.create_study(sampler=sampler)
#       ```
#       See `example.py <path/to/example.py>` for more details.
#
# - An `Others` section that describes supplementary information about the package such as the paper reference or the original source code link. For example:
#
#   .. code-block:: markdown
#
#       - [Original Paper](Link/to/the/original/paper)
#       - [Source Code](Link/to/the/source/code)
