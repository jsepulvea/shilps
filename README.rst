SHILPS - software-hardware-in-the-loop for power system simulation
=======================================================================

Currently, power systems and electricity market designers need access to
standard real-world datasets and ready-to-use and accurate simulation models.
The complexity of the systems involved and the different technological and
social layers they convey make simulation a cumbersome task. Therefore, applied
papers on power systems, active distribution networks, local energy markets
resort to simplified simulation schemes, neglecting complex interactions between
components. This tool attempts to solve this problem by providing a flexible
event--based simulation environment with standarized datasets. Moreover it
provides a standard software interface for running physical simulations.

Applications:
-------------

1. Quasistationary controllers in active distribution networks.
2. Local energy market design.
3. Interdicion--based reinforcement of gas and electricity networks.
4. Reinforcement learning: control and market participation.

Installation
------------

To install this package, follow these steps:

1. **Clone the repository:**

   .. code-block:: bash

      git clone git@github.com:jsepulvea/shilps.git
      cd shilps

2. **Create and activate a virtual environment (recommended):**

   .. code-block:: bash

      python3 -m venv venv
      source venv/bin/activate

3. **Install the package in editable mode using pip:**

   .. code-block:: bash

      pip install -e .

This will install the shilps package in editable mode, allowing you to
make changes to the code and have them immediately reflected without
reinstalling the package.

Testing
-------

1. **Install testing requirements using pip:**

   .. code-block:: bash

      pip install .[testing]

2. **Test with coverage report**

   .. code-block:: bash

      pytest --cov=shilps

Design considerations:
----------------------

- The same system that is used for simulation should be able to act as a
  reinforcement learning "gym".
- Easy timeframe configuration.
- The software is built on top of three main concepts: events, processes, and
  views.

References
----------

- `2013, Steven G. Johnson, Passing Julia Callback Functions to C <https://julialang.org/blog/2013/05/callback/>`_
- `PySCIPOpt repository <https://github.com/scipopt/PySCIPOpt/tree/master>`_
