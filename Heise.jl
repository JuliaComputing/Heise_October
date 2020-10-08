### A Pluto.jl notebook ###
# v0.12.1

using Markdown
using InteractiveUtils

# ╔═╡ b7961f1c-0949-11eb-248e-c32b842623f9
using Pkg; Pkg.activate(".")

# ╔═╡ da3498b2-0949-11eb-334c-797cbc669918
using Metalhead, Flux

# ╔═╡ 7577c812-095b-11eb-398f-3105e66097d2
using Flux3D, Zygote, AbstractPlotting, Statistics

# ╔═╡ 5ae1b5c4-094e-11eb-3acc-7b0274b783de
using DifferentialEquations, Plots

# ╔═╡ 513387ec-094d-11eb-0c0d-191d7d29324e
md"# Metalhead Example"

# ╔═╡ eaa00c4e-0949-11eb-0a3c-b7f2c42a2fa0
base_path = "data"

# ╔═╡ 04480616-0974-11eb-144f-6d9dbd5d4013
md"""
We will use an image of Philip, the corgi to test out whether our trained ML models perform on new images that they haven't seen before. In other words, do they generalize sufficiently?
"""

# ╔═╡ 036e7826-0968-11eb-0057-7da246940506
philip = load(joinpath(base_path, "philip_crop.jpg"))

# ╔═╡ e17b5e06-0973-11eb-0c5a-698b6b47eff6
md"""
## Load the pretrained VGG model

This is a classic image recognition model which has been trained on the ImageNet dataset.
"""

# ╔═╡ b780087a-094a-11eb-1a1f-5b55312cb4de
vgg = VGG19()

# ╔═╡ 0d775986-094b-11eb-2ad7-9bb105303ff3
classify(vgg, philip)

# ╔═╡ 73b1123c-0973-11eb-3caa-175b67c436bc
md"""
Excellent! Our model was able to identify that the picture is of Philip, the corgi.

Although, it did also confuse it for a cardigan, but could you really blame it? Philip is one cuddly dog.
"""

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

# ╔═╡ 6c19325a-0970-11eb-07d0-45f3cc1585a4
md"""
## Visualizing Initial Guess and Target Meshes
"""

# ╔═╡ 9362e9ec-096f-11eb-08da-d33ec592b917
[load(joinpath(base_path, "src.png")); load(joinpath(base_path, "dolphin.png"))]

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

# ╔═╡ 489c3ab0-0974-11eb-34a3-7b41aeac2d1d
md"""
As discussed earlier, we have to define a metric we wish to optimise for - our "loss function".

This would help us demonstrate whether we can effectively learn the dolphin mesh using the differentiable programming primitives.
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

# ╔═╡ 9b54115c-0970-11eb-395c-c5ac12c40971
md"""
### To run the actual training loop, set ``nepochs`` to ``2000``.
"""

# ╔═╡ 9202b23e-0970-11eb-3e83-fbe85742cb8f
nepochs = 10

# ╔═╡ b5293b88-0969-11eb-32b0-db67331d7201
begin
	θ = Flux.params(_offset)
	lr = 1.
	opt2 = Momentum(lr, 0.9)
	for itr in 1:nepochs
		gs = gradient(θ) do
			loss_dolphin(_offset, src, tgt)
		end
		Flux.update!(opt2, _offset, gs[_offset])
	end
end

# ╔═╡ 55732542-0973-11eb-1810-b1abf3900966
md"""
## Visualizing the training results
"""

# ╔═╡ 81b7c6e4-0970-11eb-0e09-efc6dbcae038
[
load(joinpath(base_path, "src.png"));
load(joinpath(base_path, "src_50.png"));
load(joinpath(base_path, "src_200.png"));
load(joinpath(base_path, "src_500.png"));
load(joinpath(base_path, "src_2000.png"));
]

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
	Plots.plot(sol,label=["Velocity" "Position"])
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

	Plots.plot(plot_t,force_plot,xlabel="t",label="True Force")
	Plots.scatter!(t,force_data,label="Force Measurements")
end

# ╔═╡ 86f2b826-0954-11eb-0c5e-27345574f616
md"""
## Define Neural Network and L2 Loss

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

	Plots.plot(plot_t,force_plot,xlabel="t",label="True Force")
	Plots.plot!(plot_t,learned_force_plot,label="Predicted Force")
	Plots.scatter!(t,force_data,label="Force Measurements")
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
	Plots.plot(sol,label=["Velocity - Real" "Position - Real"])
	Plots.plot!(sol_simplified,label=["Velocity - Ideal" "Position - Ideal"])
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

	Plots.plot(plot_t,force_plot,xlabel="t",label="True Force")
	Plots.plot!(plot_t,learned_force_plot,label="Predicted Force")
	Plots.scatter!(t,force_data,label="Force Measurements")
end

# ╔═╡ 48c3dc52-095b-11eb-1b64-af6d28273fd2
md"""
This shows the trained neural network approximating the actual force function very closely.
"""

# ╔═╡ Cell order:
# ╠═b7961f1c-0949-11eb-248e-c32b842623f9
# ╟─513387ec-094d-11eb-0c0d-191d7d29324e
# ╠═eaa00c4e-0949-11eb-0a3c-b7f2c42a2fa0
# ╠═da3498b2-0949-11eb-334c-797cbc669918
# ╟─04480616-0974-11eb-144f-6d9dbd5d4013
# ╠═036e7826-0968-11eb-0057-7da246940506
# ╟─e17b5e06-0973-11eb-0c5a-698b6b47eff6
# ╠═b780087a-094a-11eb-1a1f-5b55312cb4de
# ╠═0d775986-094b-11eb-2ad7-9bb105303ff3
# ╟─73b1123c-0973-11eb-3caa-175b67c436bc
# ╟─8f3029e2-0952-11eb-019d-f970521a289d
# ╠═7577c812-095b-11eb-398f-3105e66097d2
# ╠═927b1632-095b-11eb-2dad-ddb3af677929
# ╟─6c19325a-0970-11eb-07d0-45f3cc1585a4
# ╠═9362e9ec-096f-11eb-08da-d33ec592b917
# ╟─6449dd0e-095c-11eb-27ab-3b8d91e6617d
# ╠═f03869bc-095b-11eb-3df8-f5ba609da21b
# ╟─86f1c9b6-095c-11eb-2d83-717eb77f496d
# ╟─489c3ab0-0974-11eb-34a3-7b41aeac2d1d
# ╠═c644cfc4-095b-11eb-2802-bb397ea7c651
# ╟─d7fe6b30-095b-11eb-0de2-735b8f01fae7
# ╟─9b54115c-0970-11eb-395c-c5ac12c40971
# ╠═9202b23e-0970-11eb-3e83-fbe85742cb8f
# ╠═b5293b88-0969-11eb-32b0-db67331d7201
# ╟─55732542-0973-11eb-1810-b1abf3900966
# ╠═81b7c6e4-0970-11eb-0e09-efc6dbcae038
# ╟─06ffc29e-094d-11eb-17ef-07d69ac6cbd3
# ╟─73830ad4-094d-11eb-1047-4760599caa35
# ╠═5ae1b5c4-094e-11eb-3acc-7b0274b783de
# ╟─55afbd52-0954-11eb-2cbc-0fbd99f36080
# ╠═faf906d2-0950-11eb-2a5b-a316dc540d47
# ╟─73d23cc4-0954-11eb-2b72-9d4a380c8474
# ╠═2b7e03ba-0952-11eb-35b1-4b9558e06b26
# ╟─86f2b826-0954-11eb-0c5e-27345574f616
# ╠═5d3786a8-0952-11eb-3ac6-fd4f0a601ef9
# ╟─6f581854-0955-11eb-35db-99edd4af6d06
# ╠═cf7392c8-0952-11eb-38b5-83798ad179f1
# ╟─9e31f3f6-0954-11eb-135d-69b7d06406b7
# ╠═74a5d4ae-0953-11eb-15ec-2de8ce4a95ae
# ╟─7e17f830-0955-11eb-0c30-a3f05c0b5811
# ╟─2763ea68-0954-11eb-25bc-1960797a6aff
# ╠═9eed0ebc-0953-11eb-2f1b-795859d1bfd8
# ╟─e311d2a6-0955-11eb-084b-c99c3034d928
# ╠═b6989dd6-0955-11eb-0675-c5f03090aadc
# ╟─9807880e-0956-11eb-1b79-9b400845072b
# ╠═09b6224a-0956-11eb-25c7-91f4ecc997c5
# ╟─aca68cce-0956-11eb-205d-3968a1f27474
# ╠═7e2ae42e-0956-11eb-3299-ef918b60d2f4
# ╟─48c3dc52-095b-11eb-1b64-af6d28273fd2
