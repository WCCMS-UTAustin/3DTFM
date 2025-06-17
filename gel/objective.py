from .header import *


_EX_UND = "exclude_undetectable"
OBJECTIVE_TYPES = [
    "c_metric",
    "c_metric_easy_weight",
    "c_metric_u_weight",
    "e_metric",
    "u_metric",
    "inv_metric",
    "rel_metric",
    "c_bar_metric"
]
REGULARIZATION_TYPES = [
    "tikhonov",
    "no_regularization",
    "tv",
    "tv_log",
    "tikhonov_h1_metric",
    "tikhonov_log",
    "tikhonov_full_h1_log"
]
DOMAINS = ["entire_gel", _EX_UND]

TV_EPS=1e-8


def validate_objective(
        objective_type,
        regularization_type,
        objective_domain,
        regularization_domain,
        u_weight_filename,
        apply_u_weight_to_reg
    ):
    # Parse domains, check
    for domain_name, domain in [
        ("objective", objective_domain),
        ("regularization", regularization_domain)
    ]:
        if domain not in DOMAINS:
            if domain[:len(_EX_UND)] == _EX_UND:
                try:
                    val = float(domain[len(_EX_UND):])
                except Exception as e:
                    raise ValueError(
                        f"{domain_name} domain {objective_domain} improperly "
                        "suffixed with float string"
                    )
            else:
                raise ValueError(
                    f"{objective_domain} not a valid {domain_name} domain"
                )
    # Check if we get compatible numbers when applicable
    od_prefix = objective_domain[:len(_EX_UND)]
    rd_prefix = regularization_domain[:len(_EX_UND)]
    if od_prefix == _EX_UND and rd_prefix == _EX_UND:
        if objective_domain != regularization_domain:
            raise ValueError(
                "Exclude undetectable domains must be the same at this time, "
                f"{objective_domain} and {regularization_domain} are not."
            )

    if objective_type not in OBJECTIVE_TYPES:
        raise ValueError(f"{objective_type} not a valid objective type")
    if regularization_type not in REGULARIZATION_TYPES:
        raise ValueError(
            f"{regularization_type} not a valid regularization type"
        )

    # Check have valid file, if requested
    if u_weight_filename is not None:
        if not os.path.exists(u_weight_filename):
            raise ValueError(f"{u_weight_filename} not a valid file")

    if apply_u_weight_to_reg and (u_weight_filename is None):
        raise ValueError(
            "Must give a u weight filename to apply to regularization"
        )


def parse_domain(objective_domain, regularization_domain):
    detectable_u_cutoff = None

    od_prefix = objective_domain[:len(_EX_UND)]
    rd_prefix = regularization_domain[:len(_EX_UND)]

    if _EX_UND in [od_prefix, rd_prefix]:
        # Get the cutoff in microns from the prefix
        # Validation should ensure cutoffs the same whe matching
        if od_prefix == _EX_UND:
            detectable_u_cutoff = float(objective_domain[len(_EX_UND):])
            # Only keep the name now
            objective_domain = od_prefix
        else:
            detectable_u_cutoff = float(regularization_domain[len(_EX_UND):])
        if rd_prefix == _EX_UND:
            # Only keep the name now
            regularization_domain = rd_prefix

    return objective_domain, regularization_domain, detectable_u_cutoff


def get_objective_forms(
        geo,
        objective_type,
        regularization_type,
        objective_domain,
        regularization_domain,
        kinematics_target,
        kinematics_sim,
        mod_repr,
        gamma,
        logger=None,
        u_weight_filename=None,
        apply_u_weight_to_reg=False
    ):
    """
    Returns pre-assembled forms that make up objective.
    reg_form may be None if not using a regularizer with gamma.

    Returns:
    entire_obj_form, pure_obj_form, reg_form
    """
    # Deal with debug output
    if logger is not None:
        vprint = logger.info
    else:
        vprint = do_nothing_print

    C_target = kinematics_target.C
    C_sim = kinematics_sim.C

    # Domains
    if objective_domain == "entire_gel":
        odx = geo.dx
        vprint("Using entire gel as pure objective domain.")
    elif objective_domain == _EX_UND:
        odx = geo.dx(geo.DETECTABLE_U)
        vprint("Using detectable u volume as pure objective domain.")
    else:
        raise NotImplementedError(
            f"Don't recognize objective_domain {objective_domain}"
        )

    if regularization_domain == "entire_gel":
        rdx = geo.dx
        vprint("Using entire gel as regularization domain.")
    elif regularization_domain == _EX_UND:
        rdx = geo.dx(geo.DETECTABLE_U)
        vprint("Using detectable u volume as regularization domain.")
    else:
        raise NotImplementedError(
            f"Don't recognize regularization_domain {regularization_domain}"
        )

    # Pure objective
    w = None # Sentinel
    if objective_type == "c_metric":
        pure_obj_form = inner(C_target-C_sim,C_target-C_sim)*odx
        vprint("Using C metric as pure objective")
    elif objective_type == "c_metric_easy_weight":
        pure_obj_form = (
            inner(C_target,C_target)
            * inner(C_target - C_sim, C_target - C_sim)
            * odx
        )
        vprint(r"Using C metric with C_tar:C_tar weight as pure objective")
    elif objective_type == "c_metric_u_weight":
        pure_obj_form = (
            inner(kinematics_target.u, kinematics_target.u)
            * inner(C_target - C_sim, C_target - C_sim)
            * odx
        )
        vprint(
            r"Using C metric with weight proportional to |u_tar|^2"
        )
    elif objective_type == "e_metric":
        C_err = C_target - C_sim
        two_E_err = C_err - Identity(3)
        pure_obj_form = inner(two_E_err,two_E_err)*odx
        vprint("Using E metric as pure objective")
    elif objective_type == "u_metric":
        u_err = kinematics_target.u - kinematics_sim.u
        if u_weight_filename is None:
            pure_obj_form = inner(u_err, u_err)*odx
        else:
            vprint("Detected request for u weight from file. Loading...")
            
            w = Function(kinematics_target.geo.V0)

            w_file = XDMFFile(u_weight_filename)
            w_file.read_checkpoint(w, "w", 0)

            w.set_allow_extrapolation(True)
            w = interpolate(w, kinematics_target.geo.V0)

            vprint("...done loading")

            pure_obj_form = (w*inner(u_err, u_err))*odx

        vprint("Using u metric as pure objective")
    elif objective_type == "inv_metric":
        xi_term = (inv(C_sim) * C_target) - Identity(3)
        pure_obj_form = inner(xi_term, xi_term)*odx
        vprint("Using inv metric as pure objective")
    elif objective_type == "rel_metric":
        inv_C_tar = inv(C_target)
        id_tensor = Identity(3)
        integrand = tr(
            (C_sim*(C_sim*inv_C_tar - 2*id_tensor)*inv_C_tar) + id_tensor
        )
        pure_obj_form = integrand*odx
        vprint("Using rel metric as pure objective")
    elif objective_type == "c_bar_metric":
        C_err = conditional(
            gt(kinematics_target.Ju, 0.5),
            (
                (C_sim*kinematics_sim.Ju**(-2/3))
                - (C_target*kinematics_target.Ju**(-2/3))
            ),
            0.0*Identity(3)
        )
        pure_obj_form = inner(C_err, C_err)*odx
        vprint("Using C bar metric as pure objective")
    else:
        raise NotImplementedError(
            f"Don't recognize objective_type {objective_type}"
        )
    pre_assembly = pure_obj_form

    # Regularization
    if regularization_type == "no_regularization":
        reg_form = None
        vprint("Not using a regularization with gamma")
    else:
        if regularization_type == "tikhonov":
            ga = grad(mod_repr)
            reg_form = inner(ga,ga)
            vprint("Using Tikhonov regularization with H1 seminorm")
        elif regularization_type == "tv":
            ga = grad(mod_repr)
            reg_form = sqrt(inner(ga,ga) + TV_EPS)
            vprint("Using TV regularization")
        elif regularization_type == "tv_log":
            ga = grad(mod_repr)
            reg_form = sqrt(inner(ga,ga) + TV_EPS)/mod_repr
            vprint("Using TV regularization of log")
        elif regularization_type == "tikhonov_h1_metric":
            ga = grad(mod_repr)
            reg_form = (
                mod_repr**2
                + inner(ga, ga)
            )
            vprint("Using Tikhonov regularization with H1 metric")
        elif regularization_type == "tikhonov_log":
            gaoa = grad(mod_repr)/mod_repr
            reg_form = inner(gaoa,gaoa)
            vprint("Using Tikhonov regularization of log with H1 seminorm")
        elif regularization_type == "tikhonov_full_h1_log":
            lna = ln(mod_repr)
            gaoa = grad(mod_repr)/mod_repr
            reg_form = (inner(gaoa,gaoa)+(lna**2))
            vprint("Using Tikhonov regularization of log with full H1")
        else:
            raise NotImplementedError(
                f"Don't recognize regularization_type {regularization_type}"
            )
        
        if apply_u_weight_to_reg:
            if w is None:
                raise ValueError(
                    "Must be using a u weight to apply to regularization"
                )
            vprint("Applying u weight to regularization term.")
            # Weird typing conflicts with multiplication necessitate this
            reg_form = w * reg_form

        # Weird typing conflicts with multiplication necessitate this
        reg_form = reg_form * rdx

        pre_assembly += gamma*reg_form

    return pre_assembly, pure_obj_form, reg_form


def add_objective_arguments(parser):
    parser.add_argument(
        "-g",
        type=float,
        metavar="GAMMA",
        default=0.3,
        help="regularization parameter for this stage"
    )
    parser.add_argument(
        "--u-weight",
        type=str,
        metavar="WEIGHT_FILE",
        default=None,
        help="filename with spatially-varying weight for matching term"
    )
    parser.add_argument(
        "--apply-u-weight-to-reg",
        action="store_true",
        help="applies the weight to the regularization term as well"
    )
    parser.add_argument(
        "--ot",
        metavar="OBJECTIVE_TYPE",
        type=str,
        default="u_metric",
        help="form of the matching term"
    )
    parser.add_argument(
        "--rt",
        metavar="REGULARIZATION_TYPE",
        type=str,
        default="tikhonov",
        help="form of the regularization term"
    )
    parser.add_argument(
        "--od",
        metavar="OBJECTIVE_DOMAIN",
        type=str,
        default="exclude_undetectable0.38",
        help="domain of integral in matching term"
    )
    parser.add_argument(
        "--rd",
        metavar="REGULARIZATION_DOMAIN",
        type=str,
        default="entire_gel",
        help="domain of integral in regularization term"
    )


class ObjectiveInfo:

    def __init__(self, args):
        self.gamma_num = args.g
        self.objective_type = args.ot
        self.regularization_type = args.rt
        self.objective_domain = args.od
        self.regularization_domain = args.rd
        self.u_weight_filename = args.u_weight
        self.apply_u_weight_to_reg = args.apply_u_weight_to_reg

        validate_objective(
            self.objective_type,
            self.regularization_type,
            self.objective_domain,
            self.regularization_domain,
            self.u_weight_filename,
            self.apply_u_weight_to_reg
        )

        if self.regularization_type == "no_regularization":
            self.gamma_num = 0

        # Deal with exclude undetectable case
        # Only do this now after it is printed and directory is ready
        parsed = parse_domain(
            self.objective_domain,
            self.regularization_domain
        )
        self.objective_domain = parsed[0]
        self.regularization_domain = parsed[1]
        self.detectable_u_cutoff = parsed[2]

        self.gamma = Constant(self.gamma_num)

    def __str__(self):
        items = [
            self.gamma_num,
            self.objective_type,
            self.regularization_type,
            self.objective_domain,
            self.regularization_domain,
            self.detectable_u_cutoff,
            self.u_weight_filename,
            self.apply_u_weight_to_reg
        ]
        return ", ".join([str(o) for o in items])

    def __repr__(self):
        return str(self)

    def log_info(self, logger):
        logger.info(f"Using gamma: {self.gamma_num}")
        logger.info(f"Using objective type: {self.objective_type}")
        logger.info(f"Using regularization type: {self.regularization_type}")
        logger.info(f"Using objective domain: {self.objective_domain}")
        logger.info(f"Using reg domain: {self.regularization_domain}")
        logger.info(f"Using detectable cutoff: {self.detectable_u_cutoff}")
        logger.info(f"Using u_weight_filename: {self.u_weight_filename}")
        logger.info(f"Applying u weight to reg?: {self.apply_u_weight_to_reg}")

    def safe_u_weight_filename(self):
        if self.u_weight_filename is None:
            return None
        return os.path.basename(self.u_weight_filename)

    def get_objective_forms(self, geo, kin_tar, kin_sim, mod_repr, logger):
        pre_assembly, pure_obj_form, reg_form = get_objective_forms(
            geo,
            self.objective_type,
            self.regularization_type,
            self.objective_domain,
            self.regularization_domain,
            kin_tar,
            kin_sim,
            mod_repr,
            self.gamma,
            logger=logger,
            u_weight_filename=self.u_weight_filename,
            apply_u_weight_to_reg=self.apply_u_weight_to_reg
        )
        return pre_assembly, pure_obj_form, reg_form


def debug_deriv(
    obj_hat,
    geo,
    mod_repr,
    logger
):
    logger.info(f"A Taylor test...")

    h = Function(geo.V0)
    h.vector()[:] = np.random.rand(h.vector().local_size())
    taylor_test(obj_hat, mod_repr, h)

    logger.info(f"...completed Taylor test. Expect order 2 convergence.")

