"""
Configuration objects

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml
from fiml.auxiliaries import warning


# frozen=True   dataclass attributes cannot be modified (e.g. rewriting a value)
# slots=True    new attributes outside of those defined here cannot be created
@dataclass(frozen=True, slots=True)
class Files:
    # experimental data file
    data: str | Path
    output_dir: str | Path
    # read space and time mesh data from that file
    #mesh: str | Path | None = None
    # BCs, ICs, nu function, domain size, ecc.
    # (if None, read data from the exact solution file, if provided)
    problem: str | Path | None = None
    # exact solution (if available)
    solution: str | Path | None = None
    
    def __post_init__(self):
        # frozen=True prevents the use of usual assignments self.data = ...
        # => use this horrendous dunder method
        object.__setattr__(self, 'data', Path(self.data))
        object.__setattr__(self, 'output_dir', Path(self.output_dir))
        if self.solution is not None:
            object.__setattr__(self, 'solution', Path(self.solution))


@dataclass(frozen=True, slots=True)
class Mesh:
    dx: float
    # leave empty if the problem is steady
    dt: float | None = None
    
    def __post_init__(self):
        if self.dx < 0:
            raise ValueError('Mesh element size cannot be negative!')
        if self.dt is None:
            warning('No time step provided, the problem will be treated as steady')
        elif self.dt < 0:
            raise ValueError('Time step cannot be negative!')
        


@dataclass(frozen=True, slots=True)
class Beta:
    type: str
    num_RBF: int | None = None
    # for Gaussian RBFs
    shape_factor: float | None = None
    # for augmented RBFs, with polynomial terms
    poly_degrees_bases: tuple[int] | None = None
    # for beta with Bx and Bt: number of space and time basis functions
    num_basis_functions_x: int | None = None
    num_basis_functions_t: int | None = None
    beta_ref: float = 1.0
    beta_min: float = 0.01


@dataclass(frozen=True, slots=True)
class Optimisation:
    optimiser: str = 'ADAM'
    learning_rate: float = 1e-3
    max_iterations: int = 10_000


@dataclass(frozen=True, slots=True)
class RunOptions:
    # for unsteady problems
    show_videos: bool = False
    run_parallel: bool = False
    # max number of workers in the parallel execution pool (one job each)
    parallel_workers: int = 4
    # check all possible outputs (used during debugging)
    check_outputs: bool = True
    # skip all videos, plots and
    batch_mode: bool = False
    plot: bool = True
    save_experiments: bool = False
    # time instant to plot (default is last one)
    idx_time_plot: int = -1
    # whether to save the output to a FIML pickled object
    save_FIML_object: bool = True
    
    def __post_init__(self):
        if self.batch_mode:
            warning('BATCH mode selected, turning off all plot flags...')
            object.__setattr__(self, 'show_videos', False)
            object.__setattr__(self, 'check_outputs', False)
            object.__setattr__(self, 'plot', False)


@dataclass(frozen=True, slots=True)
class Analysis:
    variance_prior: float | str
    variance_observational: float | str
    files: Files
    # create a new Mesh instance if no value is supplied
    mesh: Mesh = field(default_factory=Mesh)
    beta: Beta = field(default_factory=Beta)
    optimisation: Optimisation = field(default_factory=Optimisation)
    options: RunOptions = field(default_factory=RunOptions)
    steady: bool = True
    regularisation_weight: float = 1
    num_samples_prior: int = 100
    num_samples_posterior: int = 100
    seed: int | None = None
    hessian: str = 'full'

    def __post__init(self):
        if self.num_samples_prior <= 0:
            raise ValueError('The number of prior samples must be > 0')
        if self.num_samples_posterior <= 0:
            warning('Number of posterior samples set to 0, skipping posterior sampling...')

    # declared as a class method => takes cls in input, not self
    # ('cls' referes to the class in itself, as a blueprint, not to the instance,
    # like 'self' does)
    @classmethod
    def from_dict(cls, input_data: dict) -> 'Analysis':
        beta_data = dict(input_data.get('beta') or {})
        # for safety, convert YAML lists to tuples
        if 'poly_degrees_bases' in beta_data:
            beta_data['poly_degrees_bases'] = tuple(beta_data['poly_degrees_bases'])
        if 'solution_limits' in beta_data:
            beta_data['solution_limits'] = tuple(beta_data['solution_limits'])

        # groups of options present also in the YAML file
        yaml_groups = {'files', 'mesh', 'beta', 'optimisation', 'options'}
        # other groups
        other_other = {key: value for key, value in input_data.items() if key not in yaml_groups}
        return cls(files=Files(**input_data['files']),
                   mesh=Mesh(**(input_data.get('mesh') or {})),
                   beta=Beta(**beta_data),
                   optimisation=Optimisation(**(input_data.get('optimisation') or {})),
                   options=RunOptions(**(input_data.get('options') or {})),
                   **other_other,
                   )
    @classmethod
    def from_yaml(cls, input_file: str | Path) -> 'Analysis':
        with open(input_file) as file:
            input_data = yaml.safe_load(file)
        return cls.from_dict(input_data)

    # convert dataclass to a dictionary
    def to_dict(self) -> dict:
        return asdict(self)
    
    # write datalcass options to yaml (for reproducibility)
    def to_yaml(self, cfg_file: str | Path) -> None:
        # convert all Path objects to strings
        yaml_data = self.to_dict()
        for file in yaml_data['files']:
            if yaml_data['files'][file] is not None:
                yaml_data['files'][file] = yaml_data['files'][file].as_posix()
        with open(cfg_file, 'w') as file:
            # (keep original key order to make YAML file easier to compare)
            yaml.safe_dump(yaml_data, file, sort_keys=False)
