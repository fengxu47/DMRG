#refrence: https://www.tensors.net/j-dmrg

"""
doApplyMPO: function for applying MPO to state
"""
#  ............psi............       M1=M2
#   |      |         |      |
#   L      M1        M2     R

function doApplyMPO(psi_In,L,M,R)

    psi = reshape(psi_In, size(L,3),size(M,4),size(M,4),size(R,3))
    @tensor psi_out[:] := psi[1,3,5,7]*L[2,-1,1]*M[2,4,-2,3]*M[4,6,-3,5]*R[6,-4,7]
    return reshape(psi_out,size(L,3)*size(M,4)*size(M,4)*size(R,3))
end



function doDMRG_MPO(A,ML,M,MR,chi; numsweeps = 10, dispon = 2, updateon = true)
    # left-to-right 'warmup', put MPS in right orthogonal form
    chid = size(M,3) #physical dimension
    Nsites = length(A) 
    L = Array{Array,1}(undef,Nsites); L[1] = ML  
    R = Array{Array,1}(undef,Nsites); R[Nsites] = MR
    for p = 1:Nsites-1
        chil = size(A[p],1) #left bond dimension
        chir = size(A[p],3) #right bond dimension
        F = svd(reshape(A[p],chil*chid,chir)) 
        A[p] = reshape(F.U,chil,chid,chir) #left-canonization
        SVt = diagm(0 => F.S) * F.Vt ./ norm(F.S)
        @tensor A[p+1][:] := SVt[-1,1] * A[p+1][1,-2,-3] 
        @tensor L[p+1][:] := L[p][2,1,4] * M[2,-1,3,5] * A[p][4,5,-3] * conj(A[p])[1,3,-2]
        
    end
    #from left to right canonization until to the last site
    chil = size(A[Nsites],1)
    chir = size(A[Nsites],3)
    F = svd(reshape(A[Nsites],chil*chid,chir))
    A[Nsites] = reshape(F.U,chil,chid,chir)
    sWeight = Array{Array,1}(undef,Nsites+1)
    sWeight[Nsites+1] = (diagm(0 => F.S)*F.Vt) ./ norm(F.S)
    
    Ekeep = zeros(0)
    B = Array{Array,1}(undef,Nsites)

    

    for k = 1:numsweeps+1

        if k == numsweeps+1
            # final sweep is only for orthogonalization (disable updates)
            updateon = false
            dispon = 0
        end

        # Optimization sweep: right-to-left  
        for p = Nsites-1:-1:1

            # two-site update
            chil = size(A[p],1) 
            chir = size(A[p+1],3)
            @tensor psi_temp[:] := A[p][-1,-2,1] * A[p+1][1,-3,2] * sWeight[p+2][2,-4]
            psiGround = reshape(psi_temp, chil*chid^2*chir)
            
            ##### Recast the `doApplyMPO` function as a 
            doApplyMPOClosed = LinearMap(psi_In -> doApplyMPO(psi_In,L[p],M,R[p+1]), chil*chid^2*chir;
             ismutating=false, issymmetric=true, ishermitian=true, isposdef=false) #d^N dimension of psiIn

            if updateon
                Entemp, psiGround = eigs(doApplyMPOClosed; nev=1, tol=1e-10, which=:SR, maxiter=300)
                #psiGround, Entemp = eigLanczos(psiGround,doApplyMPO,(L[p],M,M,R[p+1]); maxit = maxit, krydim = krydim) 
                push!(Ekeep,Entemp[1])
            end
            #decompose the optimal psiGround into A*B*sWeight
            F = svd(reshape(psiGround,chil*chid,chid*chir))
            chitemp = min(length(F.S),chi)
            A[p] = reshape(F.U[:,1:chitemp],chil,chid,chitemp)
            sWeight[p+1] = diagm(0 => (F.S[1:chitemp] ./ norm(F.S[1:chitemp])))
            B[p+1] = reshape(F.Vt[1:chitemp,:],chitemp,chid,chir)

            # new block Hamiltonian <=> next R
            @tensor R[p][:] := M[-1,2,3,5] * R[p+1][2,1,4] * B[p+1][-3,5,4] * conj(B[p+1])[-2,3,1]
            
            # display energy
            if dispon == 2
                @printf "Sweep: %d of %d, Loc: %d, Energy: %0.10f \n" k numsweeps p Ekeep[end]
            end
        end

        # left boundary tensor
        chil = size(A[1],1)
        chir = size(A[1],3)
        @tensor  A_temp[:] := A[1][-1,-2,1] * sWeight[2][1,-3]
        F = svd(reshape(A_temp, chil, chid*chir))
        B[1] = reshape(F.Vt,chil,chid,chir)
        sWeight[1] = F.U*diagm(0 => F.S)./norm(F.S)

        # Optimization sweep: left-to-right
        for p = 1:Nsites-1

            # two-site update
            chil = size(B[p],1)
            chir = size(B[p+1],3)
            @tensor psi_temp[:] := sWeight[p][-1,1]*B[p][1,-2,2]*B[p+1][2,-3,-4]
            psiGround = reshape(psi_temp, chil * chid^2 * chir)

            ##### Recast the `doApplyMPO` function as a map
            doApplyMPOClosed = LinearMap(psi_In -> doApplyMPO(psi_In,L[p],M,R[p+1]), chil*chid^2*chir;
             ismutating=false, issymmetric=true, ishermitian=true, isposdef=false) 
            if updateon
                Entemp, psiGround = eigs(doApplyMPOClosed; nev=1, tol=1e-10, which=:SR, maxiter=300)
               
                push!(Ekeep,Entemp[1])
            end
            F = svd(reshape(psiGround,chil*chid,chid*chir))
            chitemp = min(length(F.S),chi)
            A[p] = reshape(F.U[:,1:chitemp],chil,chid,chitemp)
            sWeight[p+1] = diagm(0 => (F.S[1:chitemp]./norm(F.S[1:chitemp])))
            B[p+1] = reshape(F.Vt[1:chitemp,:],chitemp,chid,chir)

            # new block Hamiltonian
            @tensor L[p+1][:] := L[p][2,1,4]*M[2,-1,3,5]*A[p][4,5,-3]*conj(A[p])[1,3,-2]
            # display energy
            if dispon == 2
                @printf "Sweep: %d of %d, Loc: %d, Energy: %0.10f \n" k numsweeps p Ekeep[end]
            end
        end

        # right boundary tensor
        chil = size(B[Nsites],1)
        chir = size(B[Nsites],3)
        @tensor A_temp[:] := sWeight[Nsites][-1,1] * B[Nsites][1,-2,-3] 
        F = svd(reshape(A_temp,chil*chid,chir))
        A[Nsites] = reshape(F.U,chil,chid,chir)
        sWeight[Nsites+1] = (diagm(0 => F.S)./norm(F.S))*F.Vt;

        if dispon == 1
            @printf "Sweep: %d of %d, Energy: %0.10f, Bond dim: %d\n" k numsweeps Ekeep[end] chi
        end
    end

    return Ekeep, A, sWeight, B
end


