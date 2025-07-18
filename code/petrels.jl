
function PETRELS(M::HePPCATModel,Y::Matrix{Float64},ΩY::AbstractMatrix,λ::Float64,δ::Float64,Fmeasure::Function,stats_fcn::Function,buffer=1)
    d,k = size(M.F)
    n = size(Y)[2]
    Rₜ⁺ = [δ*Matrix(I(k)) for i=1:d]
    Yrec = deepcopy(Y)
    data_idx = randperm(n)
    stats_log = [stats_fcn(M)]
    err_log = [Fmeasure(M)]
    time_log = [0.]

    tlast = 0
    for t = 1:n
#         j = data_idx[t]
        yₜ = Y[:,t]
        Ωₜ = ΩY[:,t]
        telapsed = @elapsed begin
        M,Rₜ⁺,yₜʳ = streamPETRELS!(M,yₜ,Ωₜ,Rₜ⁺,λ)
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

function streamPETRELS!(M::HePPCATModel,yₜ::Vector{Float64},Ωₜ::AbstractVector,Rₜ⁺::Vector{Matrix{Float64}},λ::Float64)
    Ωₜidx = findall(>(0), Ωₜ)
    F = copy(M.F)
    FΩₜ = F[Ωₜidx,:]
    yΩₜ = yₜ[Ωₜidx]
    zₜ = pinv(FΩₜ)*yΩₜ 
    Z = zₜ * zₜ'
    updatefparams!.(Ωₜ,Rₜ⁺,Ref(λ),Ref(Z))
     # updatefparams!.(Ωₜ,Rₜ⁺,Ref(λ),Ref(zₜ))
    F[Ωₜidx,:] = hcat(updatef!.(eachrow(FΩₜ),yΩₜ,Rₜ⁺[Ωₜidx],Ref(zₜ))...)'
    Fhat = svd(F)
    M.U .= Fhat.U
    M.λ .= Fhat.S.^2
    M.Vt .= Fhat.Vt
    
    yₜʳ = M.F * zₜ

    return M,Rₜ⁺,yₜʳ
end

function updatefparams!(Ωₜ,Rₜ⁺,λ,Z)

     Rₜ⁺ .= Ωₜ*Z + λ*Rₜ⁺

     return Rₜ⁺
end

# function updatefparams!(Ωₜ,Rₜ⁺,λ,zₜ)

#     βₜ = 1 + 1/λ*zₜ'*Rₜ⁺*zₜ 
#     vₜ = 1/λ*Rₜ⁺*zₜ 
#     Rₜ⁺ .= 1/λ*Rₜ⁺ - Ωₜ*1/βₜ*vₜ*vₜ'

#     return Rₜ⁺
# end

function updatef!(f,yₜ,Rₜ⁺,zₜ)

    f .+= (yₜ - f'*zₜ)*inv(Rₜ⁺)*zₜ
    # f .+= (yₜ - f'*zₜ)*Rₜ⁺*zₜ
    return f
end

# function updatef!(f,yₜ,Ωₜ,Rₜ⁺,λ,zₜ,vℓ)
# #     βₜ = 1 + 1/λ*zₜ'*Rₜ⁺*zₜ 
# #     vₜ = 1/λ*Rₜ⁺*zₜ 
# #     Rₜ⁺ .= 1/λ*Rₜ⁺ - Ωₜ/vℓ*1/βₜ*vₜ*vₜ'
#     Rₜ⁺ .= Ωₜ/vℓ*zₜ*zₜ' + λ*Rₜ⁺
# #     f .+= Ωₜ/vℓ*(yₜ - f'*zₜ)*Rₜ⁺*zₜ
#     f .+= Ωₜ/vℓ*(yₜ - f'*zₜ)*inv(Rₜ⁺)*zₜ
#     return f
# end