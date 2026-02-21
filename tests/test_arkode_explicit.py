import numpy as np
import SundialsPy
import pytest

def rhs_explicit(t, y):
    return np.array([-0.5 * y[0]])

y0 = np.array([1.0])
t0 = 0.0
t1 = 0.1
expected = y0 * np.exp(-0.5 * t1)

@pytest.mark.parametrize("butcher_table", [
    SundialsPy.arkode.ButcherTable.HEUN_EULER_2_1_2,
    SundialsPy.arkode.ButcherTable.BOGACKI_SHAMPINE_4_2_3,
])
def test_arkode_explicit_methods(butcher_table):
    solver = SundialsPy.arkode.ARKodeSolver(
        system_size=1,
        explicit_fn=rhs_explicit,
        implicit_fn=None,
        butcher_table=butcher_table
    )
    solver.initialize(y0, t0, 1e-6, np.array([1e-8]))
    result = solver.solve_to(t1)
    assert np.allclose(result, expected, atol=1e-6), f"Failed for {butcher_table}" 