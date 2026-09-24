"""The two-stage fit's bookkeeping."""

from hockey.model.multi import model_structure


def test_the_fingerprint_does_not_depend_on_the_order_categories_are_named():
    """Stage one records the categories with their own walks as the caller
    typed them, and stage two as the model orders them. Before they were
    normalised, a stage one fitted with ("hits", "assists", "sog") was refused
    by its own stage two as a different model."""
    typed = model_structure(False, ("hits", "assists", "sog"))
    ordered = model_structure(False, ("assists", "sog", "hits"))
    assert typed == ordered
    assert typed[-1] == "idio_walks:assists,sog,hits"


def test_a_board_fitted_with_hits_alone_keeps_its_fingerprint():
    # Saved stage-one files carry this exact string; it must not change.
    assert model_structure(False, ("hits",))[-1] == "idio_walks:hits"
    assert model_structure(False, False) == (
        "position_means",
        "form_factor",
        "aging",
        "availability",
        "home",
    )
