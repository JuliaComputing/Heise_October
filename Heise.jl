### A Pluto.jl notebook ###
# v0.12.1

using Markdown
using InteractiveUtils

# ╔═╡ b7961f1c-0949-11eb-248e-c32b842623f9
using Pkg; Pkg.activate("/Users/dhairyagandhi/Downloads/temp/heise")

# ╔═╡ da3498b2-0949-11eb-334c-797cbc669918
using Metalhead, Flux

# ╔═╡ 5ae1b5c4-094e-11eb-3acc-7b0274b783de
using DifferentialEquations, Plots

# ╔═╡ 513387ec-094d-11eb-0c0d-191d7d29324e
md"# Metalhead Example"

# ╔═╡ eaa00c4e-0949-11eb-0a3c-b7f2c42a2fa0
begin
	base_path = "/Users/dhairyagandhi/Downloads/temp/heise"
	philip = load(joinpath(base_path, "philip_crop.jpg"))
end

# ╔═╡ b780087a-094a-11eb-1a1f-5b55312cb4de
vgg = VGG19()

# ╔═╡ 0d775986-094b-11eb-2ad7-9bb105303ff3
classify(vgg, philip)

# ╔═╡ 8f3029e2-0952-11eb-019d-f970521a289d
md"""
# Flux3D Example
"""

# ╔═╡ 927b1632-095b-11eb-2dad-ddb3af677929
begin
	dolphin = load_trimesh(joinpath(base_path, "dolphin.obj"))
	src = load_trimesh(joinpath(base_path, "sphere.obj"))
	_offset = zeros(Float32, size(get_verts_packed(src))...)
end

# ╔═╡ d01902fc-095d-11eb-1543-7f23f03befe2
save("initial_guess.png", visualize(src));

# ╔═╡ 6449dd0e-095c-11eb-27ab-3b8d91e6617d
md"""
### Normalizing the Target Mesh
"""

# ╔═╡ f03869bc-095b-11eb-3df8-f5ba609da21b
begin
	tgt = deepcopy(dolphin)
	verts = get_verts_packed(tgt)
	center = mean(verts, dims=2)
	verts = verts .- center
	scale = maximum(abs.(verts))
	verts = verts ./ scale
	tgt._verts_packed = verts
end

# ╔═╡ 86f1c9b6-095c-11eb-2d83-717eb77f496d
md"""
### Defining the Loss
"""

# ╔═╡ c644cfc4-095b-11eb-2802-bb397ea7c651
function loss_dolphin(x::AbstractArray, src::TriMesh, tgt::TriMesh)
    src = Flux3D.offset(src, x)
    loss1 = chamfer_distance(src, tgt, 5000)
    loss2 = laplacian_loss(src)
    loss3 = edge_loss(src)
    return loss1 + 0.1*loss2 + loss3
end

# ╔═╡ d7fe6b30-095b-11eb-0de2-735b8f01fae7
md"""
## Training the Model
"""

# ╔═╡ 06ffc29e-094d-11eb-17ef-07d69ac6cbd3
md"# SciML Example - Hooke's Law"

# ╔═╡ 73830ad4-094d-11eb-1047-4760599caa35
md"""
## Imperfect Spring
### $ x \prime \prime  = - k \times x + 0.1 \times \sin(x)$
"""

# ╔═╡ 55afbd52-0954-11eb-2cbc-0fbd99f36080
md"""
## Plot Actual Solution
"""

# ╔═╡ faf906d2-0950-11eb-2a5b-a316dc540d47
begin
	k = 1.0
	force(dx,x,k,t) = -k*x + 0.1sin(x)
	prob = SecondOrderODEProblem(force,1.0,0.0,(0.0,10.0),k)
	sol = solve(prob)
	plot(sol,label=["Velocity" "Position"])
end

# ╔═╡ 73d23cc4-0954-11eb-2b72-9d4a380c8474
md"""
## Sampling Datapoints from Actual Solution
"""

# ╔═╡ 2b7e03ba-0952-11eb-35b1-4b9558e06b26
begin
	t = 0:0.001:1.0
	plot_t = 0:0.01:10
	data_plot = sol(plot_t)
	positions_plot = [state[2] for state in data_plot]
	force_plot = [force(state[1],state[2],k,t) for state in data_plot]

	# Generate the dataset
	t = 0:3.3:10
	dataset = sol(t)
	position_data = [state[2] for state in sol(t)]
	force_data = [force(state[1],state[2],k,t) for state in sol(t)]

	plot(plot_t,force_plot,xlabel="t",label="True Force")
	scatter!(t,force_data,label="Force Measurements")
end

# ╔═╡ 86f2b826-0954-11eb-0c5e-27345574f616
md"""
## Generate Neural Network and MSE loss

The neural network is trained to match the force values at every position
"""

# ╔═╡ 5d3786a8-0952-11eb-3ac6-fd4f0a601ef9
begin
	NNForce = Chain(x -> [x],
			   Dense(1,32,tanh),
			   Dense(32,1),
			   first)

	loss() = sum(abs2,NNForce(position_data[i]) - force_data[i] for i in 1:length(position_data))
end

# ╔═╡ 6f581854-0955-11eb-35db-99edd4af6d06
md"""
Train the neural network
"""

# ╔═╡ cf7392c8-0952-11eb-38b5-83798ad179f1
begin
	opt = Flux.Descent(0.01)
	data = Iterators.repeated((), 5000)
	Flux.train!(loss, Flux.params(NNForce), data, opt)
end

# ╔═╡ 9e31f3f6-0954-11eb-135d-69b7d06406b7
md"""
## Plot - Trained Model with 4 Datapoints

The trained model fits the datapoints perfectly, but does not capture the real physics
"""

# ╔═╡ 74a5d4ae-0953-11eb-15ec-2de8ce4a95ae
begin
	learned_force_plot = NNForce.(positions_plot)

	plot(plot_t,force_plot,xlabel="t",label="True Force")
	plot!(plot_t,learned_force_plot,label="Predicted Force")
	scatter!(t,force_data,label="Force Measurements")
end

# ╔═╡ 7e17f830-0955-11eb-0c30-a3f05c0b5811
md"""
This shows the neural network fit the force data but doesn't match the force function well enough.

In this case, an extra loss component which combines the constraints of the ideal spring are added
"""

# ╔═╡ 2763ea68-0954-11eb-25bc-1960797a6aff
md"""
# Plot - Ideal Spring vs Real Spring
"""

# ╔═╡ 9eed0ebc-0953-11eb-2f1b-795859d1bfd8
begin
	force2(dx,x,k,t) = -k*x
	prob_simplified = SecondOrderODEProblem(force2,1.0,0.0,(0.0,10.0),k)
	sol_simplified = solve(prob_simplified)
	plot(sol,label=["Velocity - Real" "Position - Real"])
	plot!(sol_simplified,label=["Velocity - Ideal" "Position - Ideal"])
end

# ╔═╡ e311d2a6-0955-11eb-084b-c99c3034d928
md"""
Generate more data assuming ideal spring. This is done by calculating force positions at random points.
"""

# ╔═╡ b6989dd6-0955-11eb-0675-c5f03090aadc
begin
	random_positions = [2rand()-1 for i in 1:100] # random values in [-1,1]
	loss_ode() = sum(abs2,NNForce(x) - (-k*x) for x in random_positions)
	
	# Composed loss with weighted ODE loss component
	λ = 0.1
	composed_loss() = loss() + λ*loss_ode()
end

# ╔═╡ 9807880e-0956-11eb-1b79-9b400845072b
md"""
## Train Neural Network with ODE loss component
"""

# ╔═╡ 09b6224a-0956-11eb-25c7-91f4ecc997c5
let
	opt = Flux.Descent(0.01)
	data = Iterators.repeated((), 5000)
	Flux.train!(composed_loss, Flux.params(NNForce), data, opt)
end

# ╔═╡ aca68cce-0956-11eb-205d-3968a1f27474
md"""
## Plot - Trained Neural Network vs Actual Force
"""

# ╔═╡ 7e2ae42e-0956-11eb-3299-ef918b60d2f4
let
	learned_force_plot = NNForce.(positions_plot)

	plot(plot_t,force_plot,xlabel="t",label="True Force")
	plot!(plot_t,learned_force_plot,label="Predicted Force")
	scatter!(t,force_data,label="Force Measurements")
end

# ╔═╡ 48c3dc52-095b-11eb-1b64-af6d28273fd2
md"""
This shows the trained neural network approximating the actual force function very closely.
"""

# ╔═╡ 7577c812-095b-11eb-398f-3105e66097d2
using Flux3D, AbstractPlotting, GLMakie, Statistics

# ╔═╡ f1f7e1c8-095c-11eb-36c0-bf6468b09a41
using GLMakie

# ╔═╡ Cell order:
# ╠═b7961f1c-0949-11eb-248e-c32b842623f9
# ╟─513387ec-094d-11eb-0c0d-191d7d29324e
# ╠═da3498b2-0949-11eb-334c-797cbc669918
# ╠═eaa00c4e-0949-11eb-0a3c-b7f2c42a2fa0
# ╠═b780087a-094a-11eb-1a1f-5b55312cb4de
# ╠═0d775986-094b-11eb-2ad7-9bb105303ff3
# ╟─8f3029e2-0952-11eb-019d-f970521a289d
# ╠═7577c812-095b-11eb-398f-3105e66097d2
# ╠═927b1632-095b-11eb-2dad-ddb3af677929
# ╠═f1f7e1c8-095c-11eb-36c0-bf6468b09a41
# ╠═d01902fc-095d-11eb-1543-7f23f03befe2
# ╟─6449dd0e-095c-11eb-27ab-3b8d91e6617d
# ╠═f03869bc-095b-11eb-3df8-f5ba609da21b
# ╟─86f1c9b6-095c-11eb-2d83-717eb77f496d
# ╠═c644cfc4-095b-11eb-2802-bb397ea7c651
# ╟─d7fe6b30-095b-11eb-0de2-735b8f01fae7
# ╟─06ffc29e-094d-11eb-17ef-07d69ac6cbd3
# ╟─73830ad4-094d-11eb-1047-4760599caa35
# ╠═5ae1b5c4-094e-11eb-3acc-7b0274b783de
# ╟─55afbd52-0954-11eb-2cbc-0fbd99f36080
# ╠═faf906d2-0950-11eb-2a5b-a316dc540d47
# ╟─73d23cc4-0954-11eb-2b72-9d4a380c8474
# ╠═2b7e03ba-0952-11eb-35b1-4b9558e06b26
# ╠═86f2b826-0954-11eb-0c5e-27345574f616
# ╠═5d3786a8-0952-11eb-3ac6-fd4f0a601ef9
# ╟─6f581854-0955-11eb-35db-99edd4af6d06
# ╠═cf7392c8-0952-11eb-38b5-83798ad179f1
# ╟─9e31f3f6-0954-11eb-135d-69b7d06406b7
# ╠═74a5d4ae-0953-11eb-15ec-2de8ce4a95ae
# ╟─7e17f830-0955-11eb-0c30-a3f05c0b5811
# ╟─2763ea68-0954-11eb-25bc-1960797a6aff
# ╟─9eed0ebc-0953-11eb-2f1b-795859d1bfd8
# ╟─e311d2a6-0955-11eb-084b-c99c3034d928
# ╠═b6989dd6-0955-11eb-0675-c5f03090aadc
# ╟─9807880e-0956-11eb-1b79-9b400845072b
# ╠═09b6224a-0956-11eb-25c7-91f4ecc997c5
# ╟─aca68cce-0956-11eb-205d-3968a1f27474
# ╠═7e2ae42e-0956-11eb-3299-ef918b60d2f4
# ╟─48c3dc52-095b-11eb-1b64-af6d28273fd2
