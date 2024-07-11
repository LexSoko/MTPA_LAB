
#import Pkg; Pkg.add("Meshes")
#Pkg.add("CSV")
#Pkg.add("GLMakie")
#Pkg.add("MeshViz")
#Pkg.add("ProgressBars")
#Pkg.add("Images")
#Pkg.add("DataFrames")
using GLMakie
using CSV
using ProgressBars
using Images
using DataFrames
using Meshes, MeshViz
#using GeometryBasics
path = "/Au(111) k-space/"
Data = DataFrame(CSV.File("Python/Electronthing/Au(111) k-space/Sum/08_k_PE50_FoV4c6_EScan_ap1750_HIS_Au111 copy.DAT",delim="\t"))
img = load("Python/Electronthing/Au(111) k-space/Sum/08_k_PE50_FoV4c6_EScan_ap1750_HIS_Au111_AV_001.TIF")
Data2 = DataFrame(CSV.File("Python/Electronthing/Au(111) UPS copy.DAT",delim="\t"))

fig2 = Figure(resolution=(1000,500))
ax = GLMakie.Axis(fig2[1,1],xlabel = "binding Energy / eV", xminorticksvisible = true, xticks = -24:1:0)
Energy =  Data2[!,1] .- 21.218750  
wtf = Data2[!,2]
lines!(Energy,wtf, label = "Au(111)")
hideydecorations!(ax, ticks = false)
#vlines!([5.53])
axislegend(ax, position=:rt)
save("wtf.png", fig2)
display(fig2)

function band(Data; savename= "band", e_slice = 0.10)
    fig = Figure(size = (1024,1024))
    fontsize_theme = Theme(fontsize = 40)
    set_theme!(fontsize_theme)
    dE = 21.218750 .- Data[!,"Energy"]
    dE = -dE
    img_n = Data[!,"img_name"]
    bandstruct = zeros(Float64,1024,1024,size(img_n,1))
    
    for (index,img) in tqdm(enumerate(img_n))
        bandstruct[:,:,index] = load("Python/Electronthing/Au(111) k-space/Sum/$img")
    end
    
    plane = zeros(Float64, 1024,size(img_n,1))
    for i in 1:size(img_n,1)
        for j in 1:1024
            plane[j,i] = bandstruct[j,1025-j,i]
            
        end
    end
    
    k = 0.00504 .* (-512:1:512)
    
  
    lines = [GLMakie.Point3f([k[1],k[end], minimum(dE)]),GLMakie.Point3f([k[1],k[end], maximum(dE)]),GLMakie.Point3f([k[end],k[1],maximum(dE)]),GLMakie.Point3f([k[end],k[1],minimum(dE)]),GLMakie.Point3f([k[1],k[end], minimum(dE)]) ]
    x = k
    y = k
    z = dE 
    
    fig, ax, _ = GLMakie.volume(x, y, z,bandstruct,dims=3,colormap = :plasma,colorrange = (minimum(bandstruct), maximum(bandstruct)),
            figure = (; resolution = (1400,1400), size  = (1400,1400)),  
            axis=(; type=Axis3, perspectiveness = 0.5,  azimuth = 7.19, elevation = 0.57,  
            xlabel=L"k_x / Å^{-1}", ylabel= L"k_y / Å^{-1}", zlabel="binding Energy / eV",xlabelsize= 50,ylabelsize= 50 ,zlabelsize = 50))
            
    lines!(ax, lines, color = :red, linewidth = 2)
    save("band_$savename.png", fig)
    
    fig2 = Figure(resolution = (1000,1000))
    ax2 = GLMakie.Axis(fig2[1,1], aspect = 1, xlabel = L"k_x / Å^{-1}", ylabel = "binding Energy / eV",xlabelsize= 50,ylabelsize= 50)
    heatmap!(ax2,sqrt(2) .* k , dE, log.(plane), colormap = :plasma,colorrange = (minimum(log.(bandstruct)), maximum(log.(bandstruct))))
    save("heatmap_diagonal_$savename.png", fig2)
    
    energy_index = partialsortperm(abs.(dE .+ e_slice), 1)
    slice_E = dE[energy_index] 
    fig3 = Figure(resolution = (1000,1000))
    ax3 = GLMakie.Axis(fig3[1,1], aspect = 1, xlabel = L"k_x / Å^{-1}", ylabel = L"k_y / Å^{-1}",xlabelsize= 50,ylabelsize= 50)
    heatmap!(ax3,x,y, log.(bandstruct[:,:,energy_index]),colormap = :plasma,colorrange = (minimum(log.(bandstruct[:,:,end-3])), maximum(log.(bandstruct[:,:,end-3]))))
    text!(ax3,1.7,2,text= "$slice_E eV", color=:white, fontsize= 25)
    save("$savename"*"eslice_$slice_E.png", fig3)
    display(fig)
end

band(Data,savename="Au(111)" )