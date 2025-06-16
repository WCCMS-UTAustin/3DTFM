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
# List of formulations for which mod_repr==0 is unmodified gel
ZERO_FIX_F = [
    "beta",
    "beta_tilde"
]

MU_FF_DEFAULT = 108 # Pa


# Input validation helpers
def validate_formulation(f):
    if f not in FORMULATIONS:
        raise ValueError(f"{f} not a valid formulation")


def get_neohookean_params_from_infinitesimal_analogues(formulation, mu, nu):
    """C1, D1 from formulation, mu, nu"""
    lmbda = 2*mu*nu/(1-2*nu) # Lame' parameter

    c1 = mu/2 * 1e-6
    d1 = lmbda/2 * 1e-6

    return c1, d1


def get_neohookean_psi(
        formulation,
        kinematics,
        mu_ff,
        nu,
        mod_repr,
        **kwargs
    ):
    geo = kinematics.geo

    # Far field values
    c1, d1 = get_neohookean_params_from_infinitesimal_analogues(
        formulation,
        mu_ff,
        nu
    )

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
    return interpolate(Constant(value), geo.V0)


MOD_REPR_FIELDS = {
    "zero" : (lambda geo : get_homogeneous_field(geo, 0.0)),
    "one" : (lambda geo : get_homogeneous_field(geo, 1.0))
}


def validate_mod_repr_field(mod_repr_field):
    if mod_repr_field not in MOD_REPR_FIELDS:
        if not os.path.exists(mod_repr_field):
            raise ValueError(
                f"{mod_repr_field} not a known valid mod_repr field"
            )


def get_energy(
    kinematics,
    mu_ff,
    nu,
    mod_repr,
    B,
    T,
    formulation="beta",
    bx_T=Geometry.CELL,
    **kwargs
):
    """
    Creates FEniCS expression to compute total energy, and spatially-varying
    moduli fields for a given kinematics and given material parameters, forces.

    Formulation options:
    * alpha - has alpha field on both C1, D1
    * published - alpha only on C1

    May choose to use I1 vs I1 bar (default is latter)

    Returns:
    * Pi
    """
    # Input validation
    if formulation not in FORMULATIONS:
        raise ValueError(f"{formulation} not a valid formulation")

    geo = kinematics.geo

    psi = get_neohookean_psi(
        formulation,
        kinematics,
        mu_ff,
        nu,
        mod_repr,
        **kwargs
    )

    # Total potential energy
    Pi = (
        psi*geo.dx
        - dot(B, kinematics.u)*geo.dx
        - dot(T, kinematics.u)*geo.ds(bx_T)
    )

    return Pi


def get_pk1_stress(kinematics, psi):
    """
    Given kinematics that allow differentiation wrt. F and strain energy density
    psi, returns 1st Piola-Kirchoff stress tensor.
    """
    if not kinematics.differentiable_F:
        raise ValueError(
            "Cannot compute stress tensor without F differentiability"
        )

    P = diff(psi, kinematics.F) # 1st PK stress tensor
    return P


def get_cauchy_stress(kinematics, psi):
    """
    Given kinematics that allow differentiation wrt. F and strain energy density
    psi, returns cauchy stress tensor.
    """
    if not kinematics.differentiable_F:
        raise ValueError(
            "Cannot compute Cauchy stress without F differentiability"
        )

    P = diff(psi, kinematics.F) # 1st PK stress tensor
    cauchy_stress = P*kinematics.F.T / kinematics.Ju
    return cauchy_stress


def get_nu_for_target_k(d1c1):
    """Given a target D1/C1, finds what nu should be."""
    nu = 0.5*d1c1/(1+d1c1)
    return nu


def get_k_for_target_nu(nu):
    """Given a nu used, finds what D1/C1 is."""
    twice_nu = 2*nu
    return twice_nu / (1 - twice_nu)


class Mechanics:

    def __init__(self, kinematics, formulation, mu_ff, d1c1, mod_repr):
        self.kinematics = kinematics
        self.formulation = formulation
        self.mu_ff = mu_ff
        self.d1c1 = d1c1
        self.nu = get_nu_for_target_k(self.d1c1)
        self.mod_repr = mod_repr

    def get_psi(self, kinematics=None):
        if kinematics is None:
            kinematics = self.kinematics
        psi = get_neohookean_psi(
            self.formulation,
            kinematics,
            self.mu_ff,
            self.nu,
            self.mod_repr
        )
        return psi

    def get_pk1_stress(self):
        kinematics_probe = Kinematics(
            self.kinematics.geo,
            u=self.kinematics.u,
            differentiable_F=True
        )
        psi = self.get_psi(kinematics_probe)
        P = get_pk1_stress(kinematics_probe, psi)
        return P
    
    def get_cauchy_stress(self):
        P = self.get_pk1_stress()
        cauchy_stress = P*self.kinematics.F.T / self.kinematics.Ju
        return cauchy_stress

