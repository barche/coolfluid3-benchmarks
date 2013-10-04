# Benchmark based on the incompressible Navier-Stokes demo from FEniCS
from dolfin import *

# helper function to benchmark an assembly
def time_assembly(form):
  for i in range(10):
    result = assemble(form)
  list_timings(True)
  print '#####################################################################'
  return result

# Problem size
n = 1000

parameters["form_compiler"]["cpp_optimize"] = True
#parameters['linear_algebra_backend'] = 'Epetra'
# Empty backend available in code at https://bitbucket.org/barche/dolfin
parameters['linear_algebra_backend'] = 'Empty'

# Channel mesh
height = 2.
length = 10. + DOLFIN_EPS
mesh = RectangleMesh(0., 0., length, height, n, n)

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "Lagrange", 2)
Q = FunctionSpace(mesh, "Lagrange", 1)

# Define trial and test functions
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

# Set parameter values
dt = 0.1
nu = 1.

# Create functions
u0 = Function(V)
u1 = Function(V)
p1 = Function(Q)

# Define coefficients
k = Constant(dt)

# Tentative velocity step
F1 = (1/k)*inner(u - u0, v)*dx + inner(grad(u0)*u0, v)*dx + \
     nu*inner(grad(u), grad(v))*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Pressure update
a2 = inner(grad(p), grad(q))*dx
L2 = -(1/k)*div(u1)*q*dx

# Velocity update
a3 = (1/k)*inner(u, v)*dx
L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx

list_timings(True)

# Time the assemblies
print '################ Auxiliary matrix assembly ############################'
time_assembly(a1)
print '################ Pressure matrix assembly #############################'
time_assembly(a2)
print '################ Correction matrix assembly ###########################'
time_assembly(a3)
print '################ Auxiliary RHS assembly ###############################'
time_assembly(L1)
print '################ Pressure RHS assembly ################################'
time_assembly(L2)
print '################ Correction RHS assembly ##############################'
time_assembly(L3)

