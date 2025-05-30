from functools import partial

import math
from typing import ClassVar, Optional, Tuple, Union
import jax

from jax import ensure_compile_time_eval
from jax.typing import DTypeLike

import jax.numpy as jnp
import dataclasses
from pprint import pformat
from jax.tree_util import register_pytree_node, tree_leaves

#methods from PMWD: Yin Li et al. 

def pytree_dataclass(cls, aux_fields=None, aux_invert=False, **kwargs):
    """Register python dataclasses as custom pytree nodes.

    Also added are methods that return children and aux_data iterators, and pretty
    string representation, and a method that replace fields with changes.

    Parameters
    ----------
    cls : type
        Class to be registered, not a python dataclass yet.
    aux_fields : str, sequence of str, or Ellipsis, optional
        Pytree aux_data fields. Default is none; unrecognized ones are ignored;
        ``Ellipsis`` uses all.
    aux_invert : bool, optional
        Whether to invert ``aux_fields`` selections, convenient when most but not all
        fields are aux_data.
    **kwargs
        Keyword arguments to be passed to python dataclass decorator.

    Returns
    -------
    cls : type
        Registered dataclass.

    Raises
    ------
    TypeError
        If cls is already a python dataclass.

    .. _Augmented dataclass for JAX pytree:
        https://gist.github.com/odashi/813810a5bc06724ea3643456f8d3942d

    .. _flax.struct package — Flax documentation:
        https://flax.readthedocs.io/en/latest/flax.struct.html

    .. _JAX Issue #2371:
        https://github.com/google/jax/issues/2371

    """
    if dataclasses.is_dataclass(cls):
        raise TypeError('cls cannot already be a dataclass')
    cls = dataclasses.dataclass(cls, **kwargs)

    if aux_fields is None:
        aux_fields = ()
    elif isinstance(aux_fields, str):
        aux_fields = (aux_fields,)
    elif aux_fields is Ellipsis:
        aux_fields = [field.name for field in dataclasses.fields(cls)]
    aux_data_names = [field.name for field in dataclasses.fields(cls)
                      if field.name in aux_fields]
    children_names = [field.name for field in dataclasses.fields(cls)
                      if field.name not in aux_fields]

    if aux_invert:
        aux_data_names, children_names = children_names, aux_data_names

    def children(self):
        """Return an iterator over pytree children values."""
        for name, value in self.named_children():
            yield value

    def named_children(self):
        """Return an iterator over pytree children names and values."""
        for name in children_names:
            value = getattr(self, name)
            yield name, value

    def aux_data(self):
        """Return an iterator over pytree aux_data values."""
        for name, value in self.named_aux_data():
            yield value

    def named_aux_data(self):
        """Return an iterator over pytree aux_data names and values."""
        for name in aux_data_names:
            value = getattr(self, name)
            yield name, value

    cls.children = children
    cls.named_children = named_children
    cls.aux_data = aux_data
    cls.named_aux_data = named_aux_data

    def tree_flatten(obj):
        #FIXME JAX doesn't like the flatten function to return iterators, and somehow
        #triggered AssertionError by _closure_convert_for_avals in custom_derivatives.py
        return tuple(obj.children()), tuple(obj.aux_data())

    def tree_unflatten(aux_data, children):
        return cls(**dict(zip(children_names, children)),
                   **dict(zip(aux_data_names, aux_data)))

    register_pytree_node(cls, tree_flatten, tree_unflatten)

    def _is_transforming(self):
        """Whether dataclass fields are pytrees initialized by JAX transformations.

        .. _Pytrees — JAX documentation:
            https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization

        .. _JAX Issue #10238:
            https://github.com/google/jax/issues/10238

        """
        def leaves_all(is_placeholder, tree):
            # similar to tree_all(tree_map(is_placeholder, tree))
            return all(is_placeholder(x) for x in tree_leaves(tree))

        # unnecessary to test for None's since they are empty pytree nodes
        return tree_leaves(self) and leaves_all(lambda x: type(x) is object, self)

    cls._is_transforming = _is_transforming

    def __str__(self):
        """Pretty string representation for python >= 3.10."""
        return pformat(self)

    cls.__str__ = __str__

    def replace(self, **changes):
        """Create a new object of the same type, replacing fields with changes."""
        return dataclasses.replace(self, **changes)

    cls.replace = replace

    return cls

    
@partial(pytree_dataclass, aux_fields=Ellipsis, frozen=True)
class Configuration:
    """Configuration parameters, "immutable" as a frozen dataclass.

    Parameters
    ----------
    ptcl_spacing : float
        Lagrangian particle grid cell size in [L].
    ptcl_grid_shape : tuple of int
        Lagrangian particle grid shape, in ``len(ptcl_grid_shape)`` spatial dimensions.
    mesh_shape : int, float, or tuple of int, optional
        Mesh shape. If an int or float, it is used as the 1D mesh to particle grid shape
        ratio, to determine the mesh shape from that of the particle grid. The mesh grid
        cannot be smaller than the particle grid (int or float values must not be
        smaller than 1) and the two grids must have the same aspect ratio.
    cosmo_dtype : DTypeLike, optional
        Float dtype for Cosmology and Configuration.
    pmid_dtype : DTypeLike, optional
        Signed integer dtype for particle or mesh grid indices.
    float_dtype : DTypeLike, optional
        Float dtype for other particle and mesh quantities.
    k_pivot_Mpc : float, optional
        Primordial scalar power spectrum pivot scale in 1/Mpc.
    T_cmb : float, optional
        CMB temperature in K.
    M : float, optional
        Mass unit defined in kg/h. Default is 1e10 M☉/h.
    L : float, optional
        Length unit defined in m/h. Default is Mpc/h.
    T : float, optional
        Time unit defined in s/h. Default is Hubble time 1/H_0 ~ 1e10 years/h ~ age of
        the Universe.
    transfer_fit : bool, optional
        Whether to use Eisenstein & Hu fit to transfer function. Default is True
        (subject to change when False is implemented).
    transfer_fit_nowiggle : bool, optional
        Whether to use non-oscillatory transfer function fit.
    transfer_lgk_min : float, optional
        Minimum transfer function wavenumber in [1/L] in log10.
    transfer_lgk_max : float, optional
        Maximum transfer function wavenumber in [1/L] in log10.
    transfer_lgk_maxstep : float, optional
        Maximum transfer function wavenumber step size in [1/L] in log10. It determines
        the number of wavenumbers ``transfer_k_num``, the actual step size
        ``transfer_lgk_step``, and the wavenumbers ``transfer_k``.
    growth_rtol : float, optional
        Relative tolerance for solving the growth ODEs.
    growth_atol : float, optional
        Absolute tolerance for solving the growth ODEs.
    growth_inistep: float, None, or 2-tuple of float or None, optional
        The initial step size for solving the growth ODEs. If None, use estimation. If a
        tuple, use the two step sizes for forward and reverse integrations,
        respectively.
    lpt_order : int, optional
        LPT order, with 1 for Zel'dovich approximation, 2 for 2LPT, and 3 for 3LPT.
    a_start : float, optional
        LPT scale factor and N-body starting time.
    a_stop : float, optional
        N-body stopping time (scale factor).
    a_lpt_maxstep : float, optional
        Maximum LPT light cone scale factor step size. It determines the number of steps
        ``a_lpt_num``, the actual step size ``a_lpt_step``, and the steps ``a_lpt``.
    a_nbody_maxstep : float, optional
        Maximum N-body time integration scale factor step size. It determines the number
        of steps ``a_nbody_num``, the actual step size ``a_nbody_step``, and the steps
        ``a_nbody``.
    symp_splits : tuple of float 2-tuples, optional
        Symplectic splitting method composition, with each 2-tuples being drift and then
        kick coefficients. Its adjoint has the same splits in reverse nested orders,
        i.e., kick and then drift. Default is the Newton-Störmer-Verlet-leapfrog method.
    chunk_size : int, optional
        Chunk size to split particles in batches in scatter and gather to save memory.

    Raises
    ------
    ValueError
        Incorrect or inconsistent parameter values.

    """

    ptcl_spacing: float
    ptcl_grid_shape: Tuple[int, ...]  # tuple[int, ...] for python >= 3.9 (PEP 585)

    mesh_shape: Union[float, Tuple[int, ...]] = 1

    cosmo_dtype: DTypeLike = jnp.float32
    pmid_dtype: DTypeLike = jnp.int16
    float_dtype: DTypeLike = jnp.float64

    k_pivot_Mpc: float = 0.05

    T_cmb: float = 2.7255  # Fixsen 2009, arXiv:0911.1955

    # constants in SI units, as class variables
    M_sun_SI: ClassVar[float] = 1.98847e30  # solar mass in kg
    Mpc_SI: ClassVar[float] = 3.0856775815e22  # Mpc in m
    H_0_SI: ClassVar[float] = 1e5 / Mpc_SI  # Hubble constant in h/s
    c_SI: ClassVar[int] = 299792458  # speed of light in m/s
    G_SI: ClassVar[float] = 6.67430e-11  # Gravitational constant in m^3/kg/s^2

    #hydro-related constants

    BOLTZMANN_SI:  ClassVar[float]  = 1.38065e-23         #joules/K - NIST 2010
    PROTONMASS_SI: ClassVar[float] = 1.67262178e-24/1000      #kg  - NIST 2010

    #for gaseous hydrogen

    PROTONMASS_BOLTZMANN_SI : float = PROTONMASS_SI/BOLTZMANN_SI

    
    # Units
    M: float = 1e10 * M_sun_SI
    L: float = Mpc_SI
    T: float = 1 / H_0_SI 
    K: float = 1 #kelvin

    # fairly certain need to include h^2 factor?
    
    transfer_fit: bool = True
    transfer_fit_nowiggle: bool = False
    transfer_lgk_min: float = -4
    transfer_lgk_max: float = 3
    transfer_lgk_maxstep: float = 1/128

    growth_rtol: Optional[float] = None
    growth_atol: Optional[float] = None
    growth_inistep: Union[float, None,
                          Tuple[Optional[float], Optional[float]]] = (1, None)

    lpt_order: int = 2

    a_start: float = 1/64
    a_stop: float = 1
    a_lpt_maxstep: float = 1/128
    a_nbody_maxstep: float = 1/64

    symp_splits: Tuple[Tuple[float, float], ...] = ((0, 0.5), (1, 0.5))

    chunk_size: int = 2**24

    def __post_init__(self):
        if self._is_transforming():
            return

        if isinstance(self.mesh_shape, (int, float)):
            mesh_shape = tuple(round(s * self.mesh_shape) for s in self.ptcl_grid_shape)
            object.__setattr__(self, 'mesh_shape', mesh_shape)
        if len(self.ptcl_grid_shape) != len(self.mesh_shape):
            raise ValueError('particle and mesh grid dimensions differ')
        if any(sm < sp for sp, sm in zip(self.ptcl_grid_shape, self.mesh_shape)):
            raise ValueError('mesh grid cannot be smaller than particle grid')
        if any(self.ptcl_grid_shape[0] * sm != self.mesh_shape[0] * sp
               for sp, sm in zip(self.ptcl_grid_shape[1:], self.mesh_shape[1:])):
            raise ValueError('particle and mesh grid aspect ratios differ')

        object.__setattr__(self, 'cosmo_dtype', jnp.dtype(self.cosmo_dtype))
        object.__setattr__(self, 'pmid_dtype', jnp.dtype(self.pmid_dtype))
        object.__setattr__(self, 'float_dtype', jnp.dtype(self.float_dtype))
        if not jnp.issubdtype(self.cosmo_dtype, jnp.floating):
            raise ValueError('cosmo_dtype must be floating point numbers')
        if not jnp.issubdtype(self.pmid_dtype, jnp.signedinteger):
            raise ValueError('pmid_dtype must be signed integers')
        if not jnp.issubdtype(self.float_dtype, jnp.floating):
            raise ValueError('float_dtype must be floating point numbers')

        #with jax.ensure_compile_time_eval():
        #    object.__setattr__(
        #        self,
        #        'var_tophat',
        #        TophatVar(self.transfer_k[1:], lowring=True, backend='jax'),
         #   )

        # ~ 1.5e-8 for float64, 3.5e-4 for float32
        growth_tol = math.sqrt(jnp.finfo(self.cosmo_dtype).eps)
        if self.growth_rtol is None:
            object.__setattr__(self, 'growth_rtol', growth_tol)
        if self.growth_atol is None:
            object.__setattr__(self, 'growth_atol', growth_tol)

        if any(len(s) != 2 for s in self.symp_splits):
            raise ValueError(f'symp_splits={self.symp_splits} not supported')
        symp_splits_sum = tuple(sum(s) for s in zip(*self.symp_splits))
        if symp_splits_sum != (1, 1):
            raise ValueError(f'sum of symplectic splits = {symp_splits_sum} != (1, 1)')

        dtype = self.cosmo_dtype
        for name, value in self.named_children():
            value = tree_map(lambda x: jnp.asarray(x, dtype=dtype), value)
            object.__setattr__(self, name, value)

    @property
    def dim(self):
        """Spatial dimension."""
        return len(self.ptcl_grid_shape)

    @property
    def ptcl_cell_vol(self):
        """Lagrangian particle grid cell volume in [L^dim]."""
        return self.ptcl_spacing ** self.dim

    @property
    def ptcl_num(self):
        """Number of particles."""
        with jax.ensure_compile_time_eval():
            return jnp.array(self.ptcl_grid_shape).prod().item()

    @property
    def box_size(self):
        """Simulation box size tuple in [L]."""
        return tuple(self.ptcl_spacing * s for s in self.ptcl_grid_shape)

    @property
    def box_vol(self):
        """Simulation box volume in [L^dim]."""
        with jax.ensure_compile_time_eval():
            return jnp.array(self.box_size).prod().item()

    @property
    def cell_size(self):
        """Mesh cell size in [L]."""
        return self.ptcl_spacing * self.ptcl_grid_shape[0] / self.mesh_shape[0]

    @property
    def cell_vol(self):
        """Mesh cell volume in [L^dim]."""
        return self.cell_size ** self.dim

    @property
    def mesh_size(self):
        """Number of mesh grid points."""
        with jax.ensure_compile_time_eval():
            return jnp.array(self.mesh_shape).prod().item()

    @property
    def V(self):
        """Velocity unit as [L/T]. Default is 100 km/s."""
        return self.L / self.T

    @property
    def H_0(self):
        """Hubble constant H_0 in [1/T]."""
        return self.H_0_SI * self.T

    @property
    def c(self):
        """Speed of light in [L/T]."""
        return self.c_SI / self.V

    @property
    def G(self):
        """Gravitational constant in [L^3 / M / T^2]."""
        return self.G_SI * self.M / (self.L * self.V**2)

    @property
    def PROTONMASS_BOLTZMANN(self):
        """ratio of boltzmann mass over proton mass in [K * T^2 / L^2]"""
        #there might be a factor of h needed?
        return self.PROTONMASS_BOLTZMANN_SI / (self.T**2 / (self.L**2))

    
    @property
    def rho_crit(self):
        """Critical density in [M / L^3]."""
        return 3 * self.H_0**2 / (8 * jnp.pi * self.G)

    @property
    def transfer_k_num(self):
        """Number of transfer function wavenumbers, including a leading 0."""
        return 1 + math.ceil((self.transfer_lgk_max - self.transfer_lgk_min)
                             / self.transfer_lgk_maxstep) + 1

    @property
    def transfer_lgk_step(self):
        """Transfer function wavenumber step size in [1/L] in log10."""
        return ((self.transfer_lgk_max - self.transfer_lgk_min)
                / (self.transfer_k_num - 2))

    @property
    def transfer_k(self):
        """Transfer function wavenumbers in [1/L], of ``cosmo_dtype``."""
        k = jnp.logspace(self.transfer_lgk_min, self.transfer_lgk_max,
                         num=self.transfer_k_num - 1, dtype=self.cosmo_dtype)
        return jnp.concatenate((jnp.array([0]), k))

    @property
    def a_lpt_num(self):
        """Number of LPT light cone scale factor steps, excluding ``a_start``."""
        return math.ceil(self.a_start / self.a_lpt_maxstep)

    @property
    def a_lpt_step(self):
        """LPT light cone scale factor step size."""
        return self.a_start / self.a_lpt_num

    @property
    def a_nbody_num(self):
        """Number of N-body time integration scale factor steps, excluding ``a_start``."""
        return math.ceil((self.a_stop - self.a_start) / self.a_nbody_maxstep)

    @property
    def a_nbody_step(self):
        """N-body time integration scale factor step size."""
        return (self.a_stop - self.a_start) / self.a_nbody_num

    @property
    def a_lpt(self):
        """LPT light cone scale factor steps, including ``a_start``, of ``cosmo_dtype``."""
        return jnp.linspace(0, self.a_start, num=self.a_lpt_num+1,
                            dtype=self.cosmo_dtype)

    @property
    def a_nbody(self):
        """N-body time integration scale factor steps, including ``a_start``, of ``cosmo_dtype``."""
        return jnp.linspace(self.a_start, self.a_stop, num=1+self.a_nbody_num,
                            dtype=self.cosmo_dtype)

    @property
    def growth_a(self):
        """Growth function scale factors, for both LPT and N-body, of ``cosmo_dtype``."""
        return jnp.concatenate((self.a_lpt, self.a_nbody[1:]))

#    @property
 #   def varlin_R(self):
  #      """Linear matter overdensity variance in a top-hat window of radius R in [L], of ``cosmo_dtype``."""
  #      return self.var_tophat.y

#jax.config.update("jax_enable_x64", True)


#jnp.set_printoptions(precision=6, edgeitems=2, linewidth=128)