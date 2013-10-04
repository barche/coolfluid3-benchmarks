import coolfluid as cf
from math import pi

# Flow properties
h = 1.
nu = 0.0001
re_tau = 150.
u_tau = re_tau * nu / h
a_tau = re_tau**2*nu**2/h**3
Uc = a_tau/nu*(h**2/2.)
u_ref = 0.5*Uc

# Some shortcuts
root = cf.Core.root()
env = cf.Core.environment()

# Global configuration
env.assertion_throws = False
env.assertion_backtrace = False
env.exception_backtrace = False
env.regist_signal_handlers = False
env.log_level = 4
env.only_cpu0_writes = True

# setup a model
model = root.create_component('NavierStokes', 'cf3.solver.ModelUnsteady')
domain = model.create_domain()
physics = model.create_physics('cf3.UFEM.NavierStokesPhysics')
solver = model.create_solver('cf3.UFEM.Solver')

# Add the Navier-Stokes solver as an unsteady solver
ns_solver = solver.add_unsteady_solver('cf3.UFEM.NavierStokes')
ns_solver.options.theta = 0.5

grading = 20.
y_segs = 64

x_size = 4.*pi*h
z_size = 2.*pi*h

x_segs = 128
z_segs = 128

#tstep = 0.25*(x_size/x_segs / u_ref)
tstep = 0.05

# Generate mesh
blocks = domain.create_component('blocks', 'cf3.mesh.BlockMesh.BlockArrays')
points = blocks.create_points(dimensions = 2, nb_points = 6)
points[0]  = [0, 0.]
points[1]  = [x_size, 0.]
points[2]  = [0., h]
points[3]  = [x_size, h]
points[4]  = [0.,2.*h]
points[5]  = [x_size, 2.*h]

block_nodes = blocks.create_blocks(2)
block_nodes[0] = [0, 1, 3, 2]
block_nodes[1] = [2, 3, 5, 4]

block_subdivs = blocks.create_block_subdivisions()
block_subdivs[0] = [x_segs, y_segs]
block_subdivs[1] = block_subdivs[0]

gradings = blocks.create_block_gradings()
gradings[0] = [1., 1., grading, grading]
gradings[1] = [1., 1., 1./grading, 1./grading]

left_patch = blocks.create_patch_nb_faces(name = 'left', nb_faces = 2)
left_patch[0] = [2, 0]
left_patch[1] = [4, 2]

bottom_patch = blocks.create_patch_nb_faces(name = 'bottom', nb_faces = 1)
bottom_patch[0] = [0, 1]

top_patch = blocks.create_patch_nb_faces(name = 'top', nb_faces = 1)
top_patch[0] = [5, 4]

right_patch = blocks.create_patch_nb_faces(name = 'right', nb_faces = 2)
right_patch[0] = [1, 3]
right_patch[1] = [3, 5]

blocks.extrude_blocks(positions=[z_size], nb_segments=[z_segs], gradings=[1.])

nb_procs = cf.Core.nb_procs()
#blocks.partition_blocks(nb_partitions = 2, direction = 1)
blocks.partition_blocks(nb_partitions = 8, direction = 0)
blocks.partition_blocks(nb_partitions = 4, direction = 2)

mesh = domain.create_component('Mesh', 'cf3.mesh.Mesh')
blocks.create_mesh(mesh.uri())

create_point_region = domain.create_component('CreatePointRegion', 'cf3.mesh.actions.AddPointRegion')
create_point_region.coordinates = [0., 0., 0.]  #[x_size/2., h, z_size/2.]
create_point_region.region_name = 'center'
create_point_region.mesh = mesh
create_point_region.execute()

partitioner = domain.create_component('Partitioner', 'cf3.mesh.actions.PeriodicMeshPartitioner')
partitioner.load_balance = False
partitioner.mesh = mesh

link_horizontal = partitioner.create_link_periodic_nodes()
link_horizontal.source_region = mesh.topology.right
link_horizontal.destination_region = mesh.topology.left
link_horizontal.translation_vector = [-x_size, 0., 0.]

link_spanwise = partitioner.create_link_periodic_nodes()
link_spanwise.source_region = mesh.topology.back
link_spanwise.destination_region = mesh.topology.front
link_spanwise.translation_vector = [0., 0., -z_size]

partitioner.execute()

#domain.write_mesh(cf.URI('chan150-init.cf3mesh'))

# Physical constants
physics.density = 1.
physics.dynamic_viscosity = nu
physics.reference_velocity = u_ref

ns_solver.regions = [mesh.topology.uri()]

lss = ns_solver.LSS
lss.SolutionStrategy.Parameters.preconditioner_type = 'ML'
lss.SolutionStrategy.Parameters.PreconditionerTypes.ML.MLSettings.default_values = 'NSSA'
lss.SolutionStrategy.Parameters.PreconditionerTypes.ML.MLSettings.eigen_analysis_type = 'Anorm'
lss.SolutionStrategy.Parameters.PreconditionerTypes.ML.MLSettings.aggregation_type = 'Uncoupled'
lss.SolutionStrategy.Parameters.PreconditionerTypes.ML.MLSettings.smoother_type = 'symmetric block Gauss-Seidel'
lss.SolutionStrategy.Parameters.PreconditionerTypes.ML.MLSettings.smoother_sweeps = 2
lss.SolutionStrategy.Parameters.LinearSolverTypes.Belos.solver_type = 'Block GMRES'
lss.SolutionStrategy.Parameters.LinearSolverTypes.Belos.SolverTypes.BlockGMRES.convergence_tolerance = 1e-6
lss.SolutionStrategy.Parameters.LinearSolverTypes.Belos.SolverTypes.BlockGMRES.maximum_iterations = 2000
lss.SolutionStrategy.Parameters.LinearSolverTypes.Belos.SolverTypes.BlockGMRES.num_blocks = 100


# Initial conditions
ic_u = solver.InitialConditions.create_initial_condition(builder_name = 'cf3.UFEM.InitialConditionFunction', field_tag = 'navier_stokes_solution')
ic_u.variable_name = 'Velocity'
ic_u.regions = [mesh.topology.uri()]
ic_u.value = ['{Uc}/({h}*{h})*y*(2*{h} - y)'.format(h = h, Uc = Uc), '0', '0']

# Boundary conditions
bc_u = ns_solver.BoundaryConditions
bc_u.add_constant_bc(region_name = 'bottom', variable_name = 'Velocity').value = [0., 0., 0.]
bc_u.add_constant_bc(region_name = 'top', variable_name = 'Velocity').value = [0., 0., 0.]
# Pressure BC
ns_solver.BoundaryConditions.add_constant_bc(region_name = 'center', variable_name = 'Pressure').value = 0.

# Restarter
#restart_writer = solver.add_restart_writer()
#restart_writer.Writer.file = cf.URI('chan150-{iteration}.cf3restart')
#restart_writer.interval = 100

# Time setup
time = model.create_time()
time.time_step = tstep
time.end_time = 100.*tstep

model.simulate()
model.print_timing_tree()
