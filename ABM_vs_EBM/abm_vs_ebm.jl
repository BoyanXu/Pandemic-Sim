using Agents
using Random
using Distributions
import DrWatson: @dict
using InteractiveDynamics
using GLMakie
using Plots, StatsPlots, DataFrames
using DifferentialEquations, ParameterizedFunctions
using Plots, StatsPlots, DataFrames
using RandomNumbers.Xorshifts

include("model.jl")
model = initialize()


## Visualizeation preparation
# GLMakie.activate!() # hide
# patient_color(a) = a.is_susceptible == 1 ? :green : (a.is_infected == 1 ? :red : :blue)
# patient_shape(a) = :circle

# figure, _ = abm_plot(model; ac = patient_color, am = patient_shape, as = 10)
# figure # returning the figure displays it

## Interactive application
# parange = Dict(:α => 0.01:0.5)
# adata = [(:is_susceptible, sum), (:is_infected, sum), (:is_recovered, sum)]
# alabels = ["S", "I", "R"]
# model = initialize(;number_S, griddims)

# figure, adf, mdf = abm_data_exploration(
#     model, agent_step!, dummystep, parange;
#     ac = patient_color, am = patient_shape, as = 10,
#     adata, alabels)


## Custom training spec
number_S = 499
griddims = (50, 50)
steps = 600

## Estimate meet probability
experienment_stats = []

for i in 1:10
    model = initialize(;number_S, griddims)
    adata = [(:is_susceptible, sum), (:is_infected, sum), (:is_recovered, sum)]
    mdata = [(:collision_ratio)]
    data, data2 = run!(model, agent_step!, model_step!, steps; adata, mdata)
    transform!(data,
            :sum_is_susceptible => (x -> x /= model.nb_hosts),
            :sum_is_infected => (x -> x /= model.nb_hosts),
            :sum_is_recovered => (x -> x /= model.nb_hosts),
            renamecols=false)
    push!(experienment_stats, mean(data2.collision_ratio))
end
meet_prob_estimate = mean(experienment_stats)

##
experient_stats_data = []
repetition = 50
for i in 1:repetition
    model = initialize(; number_S, griddims, meet_prob=meet_prob_estimate)
    adata = [(:is_susceptible, sum), (:is_infected, sum), (:is_recovered, sum)]
    mdata = [(:collision_ratio)]
    data, data2 = run!(model, agent_step!, model_step!, steps; adata, mdata)
    transform!(data,
            :sum_is_susceptible => (x -> x /= model.nb_hosts),
            :sum_is_infected => (x -> x /= model.nb_hosts),
            :sum_is_recovered => (x -> x /= model.nb_hosts),
            renamecols=false)
    push!(experient_stats_data, data)
end
data = reduce(.+, experient_stats_data) ./ length(experient_stats_data)

# SIR ODE model
sir_ode = @ode_def SIRModel begin
    dS = -α*S*I
    dI = α*S*I-β*I
    dR = β*I
    end α β

init = [model.number_S / model.nb_hosts, 
        model.number_I / model.nb_hosts, 
        model.number_R / model.nb_hosts]
tspan = promote(0.0,steps)
parms = [model.α, model.β]

sir_problem  = ODEProblem(sir_ode, init, tspan, parms)
sir_solution = solve(sir_problem)


@df data Plots.plot(:step, 
                    [:sum_is_susceptible, :sum_is_infected, :sum_is_recovered],
                    label = ["S" "I" "R"],
                    color = [:green :red :blue]
                    )
                    
Plots.plot!(sir_solution,
            xlabel = "Time",
            ylabel = "Number",
            label = ["S (EBM)" "I (EBM)" "R (EBM)"],
            color = [:green :red :blue],
            linestyle = :dash)


## Visualizeation
patient_color(a) = a.is_susceptible == 1 ? :green : (a.is_infected == 1 ? :red : :blue)
patient_shape(a) = :circle

model = initialize(;number_S, griddims);
abm_video(
    "sim.mp4", model, agent_step!, model_step!;
    ac = patient_color, am = patient_shape, as = 10,
    framerate = 10, frames = 600,
    title = "Schelling's segregation model"
)