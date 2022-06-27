function SHASTA_PCA(M::HePPCATModel,Y::Matrix{Float64},ΩY::AbstractMatrix,groups::Vector{Int64},learnRateParams,
                    δ::Float64,Fmeasure::Function,stats_fcn::Function,online=false,buffer=1)

    d,k = size(M.F)
    L = length(M.v)
    n = size(Y)[2]
    
    R = [δ*Matrix(I(k)) for i=1:d]
    s = [zeros(k) for i=1:d]
    ρ = zeros(L)
    θ = zeros(L)
    
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
             M, R, s, ρ, θ, yₜʳ = streamSHASTA!(M,yₜ,Ωₜ,l,w,cf,cv,R,s,ρ,θ)
            
        end
        Yrec[:,t] = yₜʳ

        tlast += telapsed
        if(mod(t,buffer)==0)
            push!(stats_log,stats_fcn(M))
            push!(err_log, Fmeasure(M))
            push!(time_log,tlast)
            tlast = 0
        end
    end
    return M, Yrec, stats_log, err_log, time_log
end

function streamSHASTA!(M::HePPCATModel,yₜ::Vector{Float64},Ωₜ::AbstractVector,l::Int64,w::Float64,cf::Float64,cv::Float64,
                        R::Vector{Matrix{Float64}},s::Vector{Vector{Float64}},ρ::Vector{Float64},θ::Vector{Float64})
    # w = learnRateParams.w
    # cf = learnRateParams.cf
    # cv = learnRateParams.cv

    M, R, s, yₜʳ = inc_updateF!(M,yₜ,Ωₜ,l,w,cf,R,s)
    M,ρ,θ = inc_updatevl!(M,yₜ,Ωₜ,l,ρ,θ,w,cv)
    
    return M, R, s, ρ, θ, yₜʳ
end

function inc_updateF!(M,yₜ,Ωₜ,l,w,cf,R,s)
    vℓ = M.v[l]
    Ωₜidx = findall(>(0), Ωₜ)
    
    F = copy(M.F)
    FΩₜ = F[Ωₜidx,:]
    yΩₜ = yₜ[Ωₜidx]
    Mtl = inv(FΩₜ'*FΩₜ + vℓ*I)
    ztl = Mtl * (FΩₜ' * yΩₜ)
    Rₜ = ztl * ztl' / vℓ + Mtl
    
    updatefparams!.(yₜ,Ωₜ,R,s,Ref(w),Ref(Rₜ),Ref(ztl),Ref(vℓ))
    
    F[Ωₜidx,:] = hcat(updatefmm!.(eachrow(FΩₜ),yΩₜ,R[Ωₜidx],s[Ωₜidx],Ref(w),Ref(cf))...)'
    Fhat = svd(F)
    
    M.U .= Fhat.U
    M.λ .= Fhat.S.^2
    M.Vt .= Fhat.Vt
    
    yₜʳ = M.F * ztl
    
    return M, R, s, yₜʳ
    
end


function inc_updatevl!(M,yₜ,Ωₜ,groupIdx,ρ,θ,w,cv)
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

            # M.v[l] = (1-cv*w)*vl + cv*w*vtl
            M.v[l] = (1-cv)*vl + cv*vtl
        else 
            ρ[l] = (1-w)*ρ[l]
            θ[l] = (1-w)*θ[l]
        end
    end
    
    return M,ρ,θ
    
end


function updatefparams!(yₜi,Ωₜi,Ri,si,w,Rₜ,ztl,vℓ)
    Ri .= (1-w)*Ri + w*(Ωₜi * Rₜ)
    si .= (1-w)*si + w*(Ωₜi/vℓ * yₜi) * ztl
    
    return Ri,si
end


function updatefmm!(fi,yₜi,Ri,si,w,cf)
    
    fₜi = inv(Ri)*si
    fi .= (1-cf)*fi + cf*fₜi
    
    return fi
end