#DMRG code (MPO version)
#two-sites update


using Printf
using LinearAlgebra
using LinearMaps
using TensorOperations
using Arpack
using Plots

include("doDMRG.jl")

chi = 16 # maximum bond dimension
Nsites = 50 # number of lattice sites

OPTS_numsweeps = 4 # number of DMRG sweeps
OPTS_dispon = 2 # level of output display
OPTS_updateon = true # update MPS tensors

#### Define Hamiltonian MPO (quantum XX model)
chid = 2 #physical dimension
sP = sqrt(2)*[0 0;1 0] 
sM = sqrt(2)*[0 1;0 0]
sX = [0 1; 1 0] 
sY = [0 -im; im 0]
sZ = [1 0; 0 -1] 
sI = [1 0; 0 1]
M = zeros(4,4,chid,chid)
M[1,1,:,:] = sI; M[4,4,:,:] = sI
M[1,2,:,:] = sM; M[2,4,:,:] = sP
M[1,3,:,:] = sP; M[3,4,:,:] = sM

ML = reshape([1;0;0;0],4,1,1) #left MPO boundary 
#contraction between ML and M => first site MPO([SI SM SP 0])
MR = reshape([0;0;0;1],4,1,1) #right MPO boundary 
##contraction between M and MR => last site MPO(transpose([0 SP SM I]))

#### Initialize MPS tensors
A = Array{Array,1}(undef,Nsites)
A[1] = rand(1,chid,min(chi,chid))
for k = 2:Nsites
    if k<Nsites-log(chi)/log(chid)
        A[k] = rand(size(A[k-1],3),chid,min(chi,size(A[k-1],3)*chid)) 
    else
        A[k] = rand(size(A[k-1],3),chid,min(size(A[k-1],3)*chid,chid^(Nsites-k)))
    end
    #warning: if Nsites-k~60, chid^(Nsites-k) will return wrong value(too big)
    #random MPS, bond dimension growth(1*d)->(d*d^2)->(d^2*d^3)->(d^3*d^4)->...->(d^2*d)->(d*1), cutoff:chi
end

#### Do DMRG sweeps (2-site approach)
En1, A, sWeight, B = doDMRG_MPO(A,ML,M,MR,chi; numsweeps = OPTS_numsweeps, dispon = OPTS_dispon, updateon = OPTS_updateon)

#### Increase bond dim and reconverge
chi = 32
En2, A, sWeight, B = doDMRG_MPO(A,ML,M,MR,chi; numsweeps = OPTS_numsweeps, dispon = OPTS_dispon, updateon = OPTS_updateon)
#choose initial state of chi=32 to be optimal state of chi=16

#### Compare with exact results (computed from free fermions Jordan-Wigner transformation)
H = diagm(1 => ones(Nsites-1)) + diagm(-1 => ones(Nsites-1))
D = eigen(0.5*(H+H'))
EnExact = 2*sum(D.values[D.values .< 0]) #free fermion, fill energy levels from bottom to top

plot(1:length(En1),log10.(En1 .-EnExact),xlabel = "Update Step",
  ylabel = "-log10(Ground Energy Error)", label = "chi = 16", title = "DMRG for XX model")
plota = plot!(1:length(En2),log10.(En2 .-EnExact), label = "chi = 32")
savefig(plota,"file.png")

#### Compute 2-site reduced density matrices, local energy profile
hamloc = reshape(real(kron(sX,sX) + kron(sY,sY)),2,2,2,2)
Enloc = zeros(Nsites-1)
for k = 1:Nsites-1
    @tensor begin
        rhotwo[:] := A[k][1,-3,2] * conj(A[k])[1,-1,3] * A[k+1][2,-4,4] * conj(A[k+1])[3,-2,5] * sWeight[k+2][4,6] * sWeight[k+2][5,6]     
        Enloc_tem[:] := hamloc[1,2,3,4] * rhotwo[1,2,3,4]
    end
    Enloc[k] = Enloc_tem[1]
end
println(Enloc)