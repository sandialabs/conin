from conin.util import try_import

with try_import() as smoek_available:
    import smoek


def add_algebraic_constraints_to_pyomo_model(*, pgm, constraints, model, data):
    if not smoek_available:
        raise TypeError(
            f"The smoek package must be installed to use algebraic constraints."
        )

    smoek_model = smoek.model()
    for func in constraints:
        func(smoek_model, data)
    smoek_model._update_smoek_components()
    smoek.pymodel.pyomo.generate(
        model=smoek_model,
        pyomo_model=model,
        data=data,
        component_map=dict(V=model.V),
    )
    return model
