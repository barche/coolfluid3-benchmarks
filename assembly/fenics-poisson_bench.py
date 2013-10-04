"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Simplest example of computation and visualization with FEniCS.

-Laplace(u) = f on the unit square.
u = u0 on the boundary.
u0 = u = 1 + x^2 + 2y^2, f = -6.
"""

from dolfin import *

n = 64

set_log_level(DEBUG)

# GNW: Use C++ optimisation flags on  generated code
parameters["form_compiler"]["cpp_optimize"] = True

# GNW: PETSc is usually faster than Epetra
#parameters['linear_algebra_backend'] = 'PETSc'
parameters['linear_algebra_backend'] = 'Epetra'

# Create mesh and define function space
#mesh = UnitSquareMesh(n, n)
mesh = UnitCubeMesh(n, n, n)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary conditions
u0 = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')

def u0_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u0, u0_boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant('-6')
a = inner(nabla_grad(u), nabla_grad(v))*dx
L = f*v*dx

for i in range(10):
  A = assemble(a)
  b = assemble(L)

## Compute solution
#u = Function(V)
#solve(a == L, u, bc)

## Dump solution to file in VTK format
#file = File('poisson.pvd')
#file << u

list_timings()