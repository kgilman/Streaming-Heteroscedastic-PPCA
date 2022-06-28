function SHASTA_PCA2(M::HePPCATModel,Y::Matrix{Float64},ΩY::AbstractMatrix,groups::Vector{Int64},learnRateParams,
                    δ::Float64,Fmeasure::Function,stats_fcn::Function,online=false,buffer=1)
    d,k = size(M.F)
    n = size(Y)[2]
    L = length(M.v)

    R = [[zeros(k,k) for i=1:d] for l=1:L]
    s = [[zeros(k) for i=1:d] for l=1:L]
    θ = zeros(L)
    α = zeros(L)
    β = [zeros(d) for l=1:L]

    
    Yrec = deepcopy(Y)
    stats_log = [stats_fcn(M)]
    err_log = [Fmeasure(M)]
    time_log = [0.]
    tlast = 0

    w0 = learnRateParams.w
    cf = learnRateParams.cf
    cv = learnRateParams.cv


    for t = 1:n

        yₜ = Y[:,t]
        l = groups[t]
        Ωₜ = ΩY[:,t]
        telapsed = @elapsed begin

            if(!online)
                w = w0/t
            else
                w = w0
            end

            M, R, s, θ, α, β, yₜʳ = streamSHASTA2!(M,yₜ,Ωₜ,l,w,cf,cv, R, s, θ, α, β)
        end
        Yrec[:,t] = yₜʳ
        tlast += telapsed
        # println(M.v)
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

function streamSHASTA2!(M::HePPCATModel,yₜ::Vector{Float64},Ωₜ::AbstractVector,l::Int64,w::Float64,cf::Float64,cv::Float64,
                        R::Vector{Vector{Matrix{Float64}}},s::Vector{Vector{Vector{Float64}}},θ::Vector{Float64},α::Vector{Float64},β::Vector{Vector{Float64}})
    M, R, s, yₜʳ = inc_updateF2!(M,yₜ,Ωₜ,l,w,cf,R,s)
    M,θ,α,β = inc_updatevl2!(M,yₜ,Ωₜ,l,θ,α,β,R,s,w,cv)
    
    return M, R, s, θ, α, β, yₜʳ
end


function inc_updateF2!(M,yₜ,Ωₜ,groupIdx,w,cf,R,s)
    vℓ = M.v[groupIdx]
    Ωₜidx = findall(>(0), Ωₜ)
    L = length(M.v)
    F = copy(M.F)
    FΩₜ = F[Ωₜidx,:]
    yΩₜ = yₜ[Ωₜidx]
    Mtl = inv(FΩₜ'*FΩₜ + vℓ*I)
    ztl = Mtl * (FΩₜ' * yΩₜ)
    Rₜ = ztl * ztl' + vℓ*Mtl
    
    # updatefparams2!.(yₜ,Ωₜ,R,s,Ref(w),Ref(Rₜ),Ref(ztl),Ref(vℓ))
    for l=1:L
        (l == groupIdx) ? lidx = 1 : lidx = 0
        updatevparams2!.(yₜ,Ωₜ,R[l],s[l],Ref(w),Ref(Rₜ),Ref(ztl),Ref(lidx))
    end
    
    Rv = sum([1/M.v[l].*R[l][Ωₜidx] for l=1:L])
    sv = sum([1/M.v[l].*s[l][Ωₜidx] for l=1:L])
    F[Ωₜidx,:] = hcat(updatefmm2!.(eachrow(FΩₜ),yΩₜ,Rv,sv,Ref(w),Ref(cf))...)'
    Fhat = svd(F)
    
    M.U .= Fhat.U
    M.λ .= Fhat.S.^2
    M.Vt .= Fhat.Vt
    
    yₜʳ = M.F * ztl
    
    return M, R, s, yₜʳ
    # return M, R, s, yₜʳ, Mtl, ztl
    
end

function inc_updatevl2!(M,yₜ,Ωₜ,groupIdx,θ,α,β,R,s,w,cf)

    L = length(M.v)
    F = copy(M.F)
    Ωₜidx = findall(>(0), Ωₜ)

    FΩₜ = F[Ωₜidx,:]
    yΩₜ = yₜ[Ωₜidx]

    for l=1:L

        if(l == groupIdx)
            vℓ = M.v[l]

            θ[l] = (1-w)*θ[l] + w*sum(Ωₜ)
            α[l] = (1-w)*α[l] + w*norm(yΩₜ)^2

            β[l] .= (1-w)*β[l] ### O(|\Omega_t|) version
            β[l][Ωₜidx] = computevparam2.(eachrow(FΩₜ),R[l][Ωₜidx],s[l][Ωₜidx])
            ρtl = α[l] - sum(β[l])

            # println(sum(β[l]) - sum(computevparam2.(eachrow(F),R[l],s[l])))

            # ρtl = α[l] - sum(computevparam2.(eachrow(F),R[l],s[l])) ###full version

            vtl = ρtl / θ[l]
            M.v[l] = (1-cf)*vℓ + cf*vtl
        else
            θ[l] = (1-w)*θ[l] 
            α[l] = (1-w)*α[l] 
            β[l] .= (1-w)*β[l]
        end
    end
    
    return M,θ,α,β
    
end


function updatevparams2!(yₜi,Ωₜi,Rli,sli,w,Rₜ,ztl,Ωl)
    Rli .= (1-w)*Rli + w*(Ωₜi*Ωl * Rₜ)
    sli .= (1-w)*sli + w*(Ωₜi*Ωl * yₜi) * ztl
    
    return Rli,sli
end

function computevparam2(fi,Rli,sli)
    return 2*fi'*sli - fi'*(Rli*fi)
    # return fi'*sli - 0.5*fi'*(Rli*fi)
end

function updatefmm2!(fi,yₜi,Ri,si,w,cf)

    fₜi = inv(Ri)*si
    fi .= (1-cf)*fi + cf*fₜi

    
    return fi
end

