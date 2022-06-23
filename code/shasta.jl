function SHASTA_PCA(M,Y,ΩY,groups,w1,w2,wf,wᵥ,L,δ,Fmeasure,stats_fcn,online=false,buffer=1)


    ### Modes: batch = 0, online = 1

    d,k = size(M.F)
    n = size(Y)[2]
    
    R = [δ*Matrix(I(k)) for i=1:d]
    s = [zeros(k) for i=1:d]
    ρ = zeros(L)
    θ = zeros(L)
    
    Yrec = deepcopy(Y)
#     data_idx = randperm(n)
    stats_log = [stats_fcn(M)]
    err_log = [Fmeasure(M)]
    time_log = [0.]
    tlast = 0
    for t = 1:n
#         j = data_idx[t]
        yₜ = Y[:,t]
        l = groups[t]
        Ωₜ = ΩY[:,t]
        telapsed = @elapsed begin
            if(online)
                M, R, s, ρ, θ, yₜʳ = streamSHASTA!(M,yₜ,Ωₜ,l,w1,w2,wf,wᵥ,R,s,ρ,θ)
            else
                M, R, s, ρ, θ, yₜʳ = streamSHASTA!(M,yₜ,Ωₜ,l,w1/t,w2/t,wf,wᵥ,R,s,ρ,θ)
            end
        end
        Yrec[:,t] = yₜʳ

        tlast += telapsed
        if(mod(t,buffer)==0)
            push!(stats_log,stats_fcn(M))
            push!(err_log, Fmeasure(M))
#             push!(time_log,telapsed)
            push!(time_log,tlast)
            tlast = 0
        end
    end
    return M, Yrec, stats_log, err_log, time_log
end

function streamSHASTA!(M,yₜ,Ωₜ,l,w1,w2,wf,wᵥ,R,s,ρ,θ)
    M, R, s, yₜʳ = inc_updateF!(M,yₜ,Ωₜ,l,w1,wf,R,s)
    M,ρ,θ = inc_updatevl!(M,yₜ,Ωₜ,l,ρ,θ,w2,wᵥ)
    
    return M, R, s, ρ, θ, yₜʳ
end

function inc_updateF!(M,yₜ,Ωₜ,l,w,wf,R,s)
    vℓ = M.v[l]
    Ωₜidx = findall(>(0), Ωₜ)
    
    F = copy(M.F)
    FΩₜ = F[Ωₜidx,:]
    yΩₜ = yₜ[Ωₜidx]
    Mtl = inv(FΩₜ'*FΩₜ + vℓ*I)
    ztl = Mtl * (FΩₜ' * yΩₜ)
    Rₜ = ztl * ztl' / vℓ + Mtl
    
    updatefparams!.(yₜ,Ωₜ,R,s,Ref(w),Ref(Rₜ),Ref(ztl),Ref(vℓ))
    
    F[Ωₜidx,:] = hcat(updatefmm!.(eachrow(FΩₜ),yΩₜ,R[Ωₜidx],s[Ωₜidx],Ref(wf))...)'
    Fhat = svd(F)
    
    M.U .= Fhat.U
    M.λ .= Fhat.S.^2
    M.Vt .= Fhat.Vt
    
    yₜʳ = M.F * ztl
    
    return M, R, s, yₜʳ
    
end


function inc_updatevl!(M,yₜ,Ωₜ,groupIdx,ρ,θ,w,wᵥ)
    L = length(M.v)

    for l=1:L 
        if(l == groupIdx)
   
            vl = M.v[l]
            Ωₜidx = findall(>(0), Ωₜ)
            FΩₜ = M.F[Ωₜidx,:]
            Mtl = inv(FΩₜ'*FΩₜ + vl*I)
            yΩₜ = yₜ[Ωₜidx]

            ρtl =  norm(yΩₜ - FΩₜ*(Mtl*(FΩₜ'*yΩₜ)))^2 + vl*tr((FΩₜ'*FΩₜ)*Mtl)
            
            ### Update the surrogates
            ρ[l] = (1-w)*ρ[l] + w*ρtl
            θ[l] = (1-w)*θ[l] + w*sum(Ωₜ)

            vtl = ρ[l] / θ[l]

            M.v[l] = (1-wᵥ)*vl + wᵥ*vtl
        else 
            ρ[l] = (1-w)*ρ[l]
            θ[l] = (1-w)*θ[l]
        end
    end
    
    return M,ρ,θ
    
end

# function inc_updatevl!(M,yₜ,Ωₜ,l,ρ,θ,w)
#     U = M.U
#     λ = M.λ
#     vl = M.v[l]
#     L = length(M.v)
#     d, k = size(U)
#     Ωₜidx = findall(>(0), Ωₜ)
    
#     FΩₜ = M.F[Ωₜidx,:]
#     Mtl = inv(FΩₜ'*FΩₜ + vl*I)
#     yΩₜ = yₜ[Ωₜidx]

    
#     ρtl =  norm(yΩₜ - FΩₜ*(Mtl*(FΩₜ'*yΩₜ)))^2 + vl*tr((FΩₜ'*FΩₜ)*Mtl)
    
#     ### Update the surrogates
#     ρ[l] = (1-w)*ρ[l] + w*ρtl
#     θ[l] = (1-w)*θ[l] + w*sum(Ωₜ)

#     # vlt =  ρ[l] / sum(Ωₜ)
#     vtl = ρ[l] / θ[l]

#     M.v[l] = (1-w)*vl + w*vtl
    
#     return M,ρ
    
# end

nonnegative(x) = max(zero(x),x)

function updatefparams!(yₜi,Ωₜi,Ri,si,w,Rₜ,ztl,vℓ)
    Ri .= (1-w)*Ri + w*(Ωₜi * Rₜ)
    si .= (1-w)*si + w*(Ωₜi/vℓ * yₜi) * ztl
    
    return Ri,si
end

# function updatefmm!(fi,yₜi,Ωₜi,Ri,si,w,Rₜ,ztl,vℓ)
function updatefmm!(fi,yₜi,Ri,si,w)
    
#     Ri .= (1-w)*Ri + w*(Ωₜi * Rₜ)
#     si .= (1-w)*si + w*(Ωₜi/vℓ * yₜi) * ztl
    

    fₜi = inv(Ri)*si
    fi .= (1-w)*fi + w*fₜi

    
    return fi
end