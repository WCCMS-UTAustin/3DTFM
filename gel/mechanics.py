r"""Implementations of material model formulations and stress tensors

# Material Model Formulations

Valid material model formulations are listed in `FORMULATIONS`, below
are the strain-energy densities they correspond to. It is supplied to
`Mechanics` in the `formulation` argument of its constructor.

## "beta"

This is the official best formulation.
`mod_repr` is $\beta(\mathbf{x}_0)$.

$$\Psi=c_1e^{\beta(\mathbf{x}_0)}\left(I_1-3-2\ln{J}\right)+D_1(\ln{J})^2$$

## "alpha"

This is the formulation described in our previous approach.
`mod_repr` is $\alpha(\mathbf{x}_0)$

$$\Psi=\alpha(\mathbf{x}_0)c_1\left(I_1-3-2\ln{J}\right)+D_1(\ln{J})^2$$

## "alpha_on_all"

This formulation models both material parameters as varying in the same
way.
`mod_repr` is $\alpha(\mathbf{x}_0)$

$$\Psi=\alpha(\mathbf{x}_0)\left[c_1\left(I_1-3-2\ln{J}\right)+D_1(\ln{J})^2\right]$$

## "exclude_all_penalty"

This formulation has only been used for testing purposes; testing
revealed that this formulation predicts nonzero stress in some
undeformed configurations, and is therefore aphysical.
`mod_repr` is $\alpha(\mathbf{x}_0)$

$$\Psi=\alpha(\mathbf{x}_0)c_1\left(I_1-3\right)-2c_1\ln{J}+D_1(\ln{J})^2$$

## "beta_tilde"

This formulation was designed to utilize the benefits of the exponential
formulation while restricting modulus prediction to lie within a
prescribed range. The range must be supplied to `kwargs` in `Mechanics`
through float entries "beta_min" $\beta_{min}$ and "beta_max"
$\beta_{max}$.
`mod_repr` is $\beta(\mathbf{x}_0)$. A helper field is defined

$$\tilde{\beta}(\mathbf{x}_0)=a\tanh\left(m\beta(\mathbf{x}_0)+b\right)+c$$

where

$$a=\frac{\beta_{max}-\beta_{min}}{2}$$
$$c=\frac{\beta_{max}+\beta_{min}}{2}$$
$$b=-\text{arctanh}\left(\frac{c}{a}\right)$$
$$m=\frac{1}{a\left(1-\left(\frac{c}{a}\right)^2\right)}$$

and then strain energy density is

$$\Psi=c_1e^{\tilde{\beta}(\mathbf{x}_0)}\left(I_1-3-2\ln{J}\right)+D_1(\ln{J})^2$$

# Modulus Representation "mod_repr"

The `mod_repr` argument to `Mechanics` is an appropriate FEniCS
function. In contrast, the `create_mod_repr` function takes a str desc
and generates an appropriate FEniCS 1st-order Lagrange scalar field on
the mesh.

If `desc` matches an option in the dict `MOD_REPR_FIELDS`, then the
field will be assembled correspondingly. Otherwise, it will be
interpreted as the path to a full-shape/write-checkpoint .xdmf file with
the field in the attribute "mod_repr"

# API
"""
from .header import *
from .geometry import *
from .kinematics import *

from ufl import tanh


FORMULATIONS = [
    "beta",
    "alpha",
    "alpha_on_all",
    "exclude_all_penalty", # Note: stress not always 0 at no deformation
    "beta_tilde"
]
"""List of str valid formulation names"""

# 
ZERO_FIX_F = [
    "beta",
    "beta_tilde"
]
"""List of formulations for which mod_repr==0 is unmodified gel

Rest have mod_repr==1 is unmodified gel.
"""

MU_FF_DEFAULT = 108 # Pa
r"""Rheometric measurement of shear modulus for $c_1=\frac{\mu}{2}$ in Pa"""


# Input validation helpers
def validate_formulation(f):
    """Raises `ValueError` if `f` str is not a valid formulation."""
    if f not in FORMULATIONS:
        raise ValueError(f"{f} not a valid formulation")


def _get_neohookean_psi(
        formulation,
        kinematics,
        mu_ff,
        nu,
        mod_repr,
        **kwargs
    ):
    geo = kinematics.geo

    # Far field values
    lmbda = 2*mu_ff*nu/(1-2*nu) # 1st Lame' parameter
    c1 = mu_ff/2 * 1e-6
    d1 = lmbda/2 * 1e-6

    I1 = kinematics.Ic

    lnJ = ln(kinematics.Ju)

    # Stored strain energy density (compressible neo-Hookean model)
    if formulation == "beta":
        psi = exp(mod_repr)*c1*(I1 - 3 - 2*lnJ) + d1*(lnJ)**2
    elif formulation == "alpha":
        psi = mod_repr*c1*(I1 - 3 - 2*lnJ) + d1*(lnJ)**2
    elif formulation == "alpha_on_all":
        psi = mod_repr*(c1*(I1 - 3 - 2*lnJ) + d1*(lnJ)**2)
    elif formulation == "exclude_all_penalty":
        psi = mod_repr*c1*(I1 - 3) - 2*c1*lnJ + d1*(lnJ)**2
    elif formulation == "beta_tilde":
        beta_max = kwargs["beta_max"]
        beta_min = kwargs["beta_min"]

        a = 0.5*(beta_max - beta_min)
        c = 0.5*(beta_max + beta_min)
        b = -np.arctanh(c/a)
        m = 1/(a*(1-((c/a)**2)))

        beta = a*tanh(m*mod_repr + b) + c

        psi = exp(beta)*c1*(I1 - 3 - 2*lnJ) + d1*(lnJ)**2
    else:
        # Assume no modification
        psi = c1*(I1 - 3) - 2*c1*lnJ + d1*(lnJ)**2

    return psi


def get_homogeneous_field(geo, value):
    """Returns FEniCS function with a homogeneous scalar value

    * `geo`: type `gel.geometry.Geometry` with mesh whereupon the scalar
    field will be defined
    * `value`: float value the field will take
    """
    return interpolate(Constant(value), geo.V0)


MOD_REPR_FIELDS = {
    "zero" : (lambda geo : get_homogeneous_field(geo, 0.0)),
    "one" : (lambda geo : get_homogeneous_field(geo, 1.0))
}
"""dict of names of pre-defined modulus representation fields to FEniCS
function factories that take in a `gel.geometry.Geometry` as a single
argument
"""


def validate_mod_repr_field(mod_repr_field):
    """Raises `ValueError` if `mod_repr_field` cannot be interpreted as
    a valid scalar field (see top of `gel.mechanics`)
    """
    if mod_repr_field not in MOD_REPR_FIELDS:
        if not os.path.exists(mod_repr_field):
            raise ValueError(
                f"{mod_repr_field} not a known valid mod_repr field"
            )


def create_mod_repr(geo, desc):
    """Returns FEniCS FE scalar function according to the description.

    * `geo`: `gel.geometry.Geometry` with the mesh and function space
    * `desc`: str description of the field, see the intro to
    `gel.mechanics`.

    Returns: FEniCS field named "ctl" due to primary use as control
    variable
    """
    if isinstance(desc, str):
        if desc in MOD_REPR_FIELDS:
            # Case lookup
            mod_repr = MOD_REPR_FIELDS[desc](geo)
        else:
            # Case read from file
            mod_repr = load_shape(geo.V0, desc, "mod_repr")
            mod_repr.set_allow_extrapolation(True)
    else:
        # Case given the function
        mod_repr = desc
    mod_repr.rename("ctl", "ctl")

    return mod_repr


def _get_nu_for_target_d1c1(d1c1):
    """Returns Poisson ratio float from D1/C1 `d1c1` float."""
    nu = 0.5*d1c1/(1+d1c1)
    return nu


def _get_d1c1_for_target_nu(nu):
    """Returns D1/C1 float from Poisson ratio `nu` float."""
    twice_nu = 2*nu
    return twice_nu / (1 - twice_nu)


class Mechanics:
    """Information on and functions to retrieve material properties"""

    def __init__(
            self,
            kinematics,
            formulation="beta",
            mu_ff=MU_FF_DEFAULT,
            d1c1=1.0,
            mod_repr=None,
            bx_T=Geometry.CELL,
            **kwargs
        ):
        r"""Object for convenient handling of mechanical quantities

        * `kinematics`: `gel.kinematics.Kinematics` object with
        displacements to use by default
        * `formulation`: str the name of the formulation (see
        `gel.mechanics`)
        * `mu_ff`: float unmodified gel shear modulus from rheometry,
        used to compute $c_1=\frac{\mu}{2}$
        * `d1c1`: float ratio $\frac{D_1}{c_1}$ encoding compressibility
        * `mod_repr`: FEniCS scalar field
        * `bx_T`: surface tag used by `kinematics.geo` (see
        `gel.geometry.Geometry`) on which to apply traction force when
        provided to subsequent function calls.
        * `kwargs`: dict additional material model formulation
        parameters, see `gel.mechanics` intro.
        """
        validate_formulation(formulation)

        self.kinematics = kinematics
        """Type `gel.kinematics.Kinematics` with deformation data with
        which to compute mechanical quantities by default
        """

        self.formulation = formulation
        """str name of the formulation from `FORMULATIONS`"""
        self.formulation_kwargs = kwargs
        """dict additional arguments to formulation, see intro to
        `gel.mechanics`
        """

        self.mu_ff = mu_ff
        """Far-field (unmodified) infinitesimal strain shear modulus"""

        self.d1c1 = d1c1
        r"""Measure of incompressibility $\frac{D_1}{c_1}$"""

        self.nu = _get_nu_for_target_d1c1(self.d1c1)
        """Infinitesimal strain Poisson ratio for unmodified gel"""

        self.bx_T = bx_T
        """Boundary on which external traction would be applied"""

        if mod_repr is None:
            mod_repr = MOD_REPR_FIELDS["zero"](kinematics.geo)

        self.mod_repr = mod_repr
        """FEniCS scalar function for modulus control/representation"""

    def get_psi(self, kinematics=None):
        """Returns FEniCS expression for strain-energy density

        Units are MPa

        * `kinematics`: `gel.kinematics.Kinematics` optionally use a
        different instance than the one originally provided to this
        object for purposes such as computing stress tensors
        """
        if kinematics is None:
            kinematics = self.kinematics

        psi = _get_neohookean_psi(
            self.formulation,
            kinematics,
            self.mu_ff,
            self.nu,
            self.mod_repr,
            **self.formulation_kwargs
        )

        return psi

    def get_energy(self, kinematics=None, B=None, T=None):
        """
        Returns FEniCS expression for total energy

        Units are pJ

        * `kinematics`: `gel.kinematics.Kinematics` optionally use a
        different instance than the one originally provided to this
        object for purposes such as computing stress tensors
        * `B`: FEniCS expression for body force in hydrogel domain, by
        default 0
        * `T`: FEniCS expression for traction force applied on `bx_T`,
        by default 0
        """
        if kinematics is None:
            kinematics = self.kinematics

        if B is None:
            B = Constant((0.0, 0.0, 0.0))

        if T is None:
            T = Constant((0.0, 0.0, 0.0))

        psi = self.get_psi(kinematics=kinematics)

        Pi = (
            psi*kinematics.geo.dx
            - dot(B, kinematics.u)*kinematics.geo.dx
            - dot(T, kinematics.u)*kinematics.geo.ds(self.bx_T)
        )

        return Pi

    def get_pk1_stress(self):
        """Returns 1st Piola-Kirchoff stress tensor FEniCS expression.

        Units are MPa

        Not ideal for computing surface tractions due to inaccuracy
        in numerical integration, see `get_nodal_traction` for that
        instead.
        """
        kinematics_probe = Kinematics(
            self.kinematics.geo,
            u=self.kinematics.u,
            differentiable_F=True
        )

        psi = self.get_psi(kinematics_probe)

        P = diff(psi, kinematics_probe.F) # 1st PK stress tensor

        return P
    
    def get_cauchy_stress(self):
        """Returns Cauchy stress tensor FEniCS expression.

        Units are MPa
        """
        P = self.get_pk1_stress()
        cauchy_stress = P*self.kinematics.F.T / self.kinematics.Ju
        return cauchy_stress

    def get_total_strain_energy(self):
        """Returns float total strain energy in units of pJ"""
        psi = self.get_psi()
        Pi = psi*self.kinematics.geo.dx
        return assemble(Pi) # In MPa*um^3 = J * 10^{-12}

    def get_nodal_traction(self, surf_marker=Geometry.CELL):
        """Returns FEniCS function with nodal surface traction forces.

        Units are MPa. Only the case for `formulation` "beta" allowed.

        * `surf_marker`: surface tag from `gel.geometry.Geometry` on
        which to compute traction force.

        Returns: a function in the vector 1st order Lagrange basis on
        the full hydrogel mesh. DoFs on the requested surface are the
        projected traction forces.

        Raises: `ValueError` is formulation not implemented
        """
        if self.formulation != "beta":
            raise ValueError(
                "Nodal traction computation only implemented for beta "
                "formulation due to requirement to manually define form"
                " of PK tensor for full accuracy"
            )

        geo = self.kinematics.geo
        kin = self.kinematics

        alpha = exp(self.mod_repr)

        F = kin.F
        J = kin.Ju
        kin.u.set_allow_extrapolation(True)

        Finv = inv(F)

        P = self.mu_ff*1e-6*(
            alpha*(F.T - Finv.T)
            + self.d1c1*ln(J)*Finv.T
        )

        Vproj = geo.V
        t = Function(Vproj)

        n = FacetNormal(geo.mesh)
        T = dot(P, n)

        dt = TrialFunction(Vproj)
        v = TestFunction(Vproj)

        L = assemble(dot(T, v)*geo.ds(surf_marker))
        A = assemble(dot(dt, v)*geo.ds(surf_marker), keep_diagonal=True)
        A.ident_zeros()

        solve(A, t.vector(), L)

        return t

