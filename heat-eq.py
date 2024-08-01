import parmed
from openmm.app import *
from openmm import *
from openmm.unit import *
from parmed.openmm.reporters import RestartReporter
from sys import stdout
from math import floor

#Equilibration Parameters
hmass = 2 #HMR enabled for longer timesteps
shortstepsize = 0.002 * picoseconds

# set temp increments for heating
bottom = 100
stride = 10
topt = 300 + stride  # needs to be + stride because of how range works
temp_heating_range = list(map(lambda x: x * kelvin, range(bottom, topt, stride)))
k_posre = 100.0
restrained_duration = 10 ** 6
unrestrained_duration = 10 ** 5 

#Production Parameters
longstepsize = 0.004 * picoseconds
pressure = 1.0 * atmospheres
temperature = 300 * kelvin
friction = 1.0/picosecond
constraint_tolerance = 1e-8
temp_interval = int(floor(restrained_duration / len(temp_heating_range)))

platform = Platform.getPlatformByName('CUDA')
platform_properties = {'Precision': 'mixed'}


# Load AMBER files
print('Loading AMBER files...', flush=True)

# inpcrd = AmberInpcrdFile('center-exact-waters.inpcrd')
# prmtop = AmberPrmtopFile('center-exact-waters.prmtop', periodicBoxVectors=inpcrd.boxVectors)

prmtop_crds = parmed.load_file('center-exact-waters.prmtop', 'center-exact-waters.inpcrd')
print('done reading system files')

system = prmtop_crds.createSystem(nonbondedMethod=PME, nonbondedCutoff=1 * nanometer,
                                  constraints=HBonds, hydrogenMass=hmass * amu,
                                  rigidWater=True)

        # includeDir='/opt/user/gromacs/gcc/9.3.0/gromacs_2023.2_cpu/share/gromacs/top')

#Initialize the Integrator
integrator = LangevinMiddleIntegrator(bottom * kelvin, friction, shortstepsize)
integrator.setConstraintTolerance(constraint_tolerance)

#Add restraints 
restraint = CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
restraint_ix = system.addForce(restraint)
restraint.addGlobalParameter('k', k_posre * kilojoules_per_mole / nanometer)
restraint.addPerParticleParameter('x0')
restraint.addPerParticleParameter('y0')
restraint.addPerParticleParameter('z0')

print('Adding restraints to CAs.')
for i, atom in enumerate(prmtop_crds.topology.atoms()):
    if i % 2000 == 0:
        print(f'Checking atom {i} to see if it needs restraint.')
    if atom.name == 'CA':
        restraint.addParticle(atom.index, prmtop_crds.positions[atom.index])

print('Initializing Simulation')
simulation = simulation.Simulation(prmtop_crds.topology, system, integrator,platform, platform_properties)
simulation.context.setPositions(prmtop_crds.positions)
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(bottom)

simulation.reporters.append(statedatareporter.StateDataReporter(stdout, 100, step=True,
                                                                potentialEnergy=True,
                                                                temperature=True))

#Running NVT
# NVT heating simulation with restraints
print('Heating')
for T in temp_heating_range:
    print(f'Temp is now {T}')
    simulation.integrator.setTemperature(T)
    simulation.step(temp_interval)

#Run NPT
print('Running NPT')
barostat = openmm.MonteCarloBarostat(pressure, temperature)
system.addForce(barostat)
system.removeForce(restraint_ix)
simulation.integrator.setStepSize(longstepsize)
simulation.context.reinitialize(preserveState=True)
simulation.step(unrestrained_duration)

State = simulation.context.getState(getPositions=True, getVelocities=True,
                                    getForces=True, getEnergy=True)

print('Serializing State')
# Serialize state
state_filename = f'state.xml'
with open(state_filename, 'w') as f:
    input = XmlSerializer.serialize(State)
    f.write(input)

# Serialize integrator
with open(f'./integrator.xml', 'w') as f:
    input = XmlSerializer.serialize(integrator)
    f.write(input)

print('Serializing System')
# Serialize system
with open(f'./system.xml','w') as f:
    input = XmlSerializer.serialize(system)
    f.write(input)
