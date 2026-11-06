import torch
from cheetah.accelerator import Drift, Quadrupole, Screen, Segment
from cheetah.particles import ParticleBeam
from lume_cheetah.actions import CheetahReadOnlyNDVariable, CheetahWritableScalarVariable, CheetahBeamPropertyVariable
from lume_cheetah.model import LUMECheetahModel
from lume_cheetah.simulator import CheetahSimulator

def create_example_model():
    """Create a LUMECheetah model with three quadrupoles, drifts, and a screen."""

    segment = Segment([
        Quadrupole(name='Q1', length=torch.tensor(0.5), k1=torch.tensor(1.0)),
        Drift(name='D1', length=torch.tensor(1.0)),
        Quadrupole(name='Q2', length=torch.tensor(0.5), k1=torch.tensor(-1.0)),
        Drift(name='D2', length=torch.tensor(1.0)),
        Quadrupole(name='Q3', length=torch.tensor(0.5), k1=torch.tensor(1.0)),
        Drift(name='D3', length=torch.tensor(1.0)),
        Screen(
            name='SCR1', resolution=(64, 64),
            pixel_size=torch.tensor([10e-4, 10e-4]),
            method='cloud-in-cell', is_active=True,
        ),
    ])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        particle_beam = ParticleBeam.from_twiss(
            beta_x=torch.tensor(1.0), beta_y=torch.tensor(1.0),
            emittance_x=torch.tensor(1e-7), emittance_y=torch.tensor(1e-7),
            num_particles=1000, energy=torch.tensor(100e6),
            total_charge=torch.tensor(1e-12),
        )
    simulator = CheetahSimulator(
        segment=segment, initial_beam_distribution=particle_beam,
    )
    variables = [
        CheetahWritableScalarVariable(
            name='Q1_k', element_name='Q1', element_attribute='k1',
        ),
        CheetahWritableScalarVariable(
            name='Q2_k', element_name='Q2', element_attribute='k1',
        ),
        CheetahWritableScalarVariable(
            name='Q3_k', element_name='Q3', element_attribute='k1',
        ),
        CheetahReadOnlyNDVariable(
            name='screen_image', element_name='SCR1',
            element_attribute='reading', shape=(64, 64),
        ),
        CheetahBeamPropertyVariable(
            name="screen_centroid_x",
            element_name="SCR1",
            beam_property="mu_x",
            unit="m"
        ),
        CheetahBeamPropertyVariable(
            name="screen_centroid_y",
            element_name="SCR1",
            beam_property="mu_y",
            unit="m"
        )
    ]

    # Create the reference model with the defined simulator and variables
    reference_model = LUMECheetahModel(simulator, action_variables=variables)
    return reference_model