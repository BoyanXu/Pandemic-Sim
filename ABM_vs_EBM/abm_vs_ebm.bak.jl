using Agents
using Random
using Distributions
import DrWatson: @dict
using InteractiveDynamics
using GLMakie
using Plots, StatsPlots, DataFrames
using DifferentialEquations, ParameterizedFunctions


@agent Patient GridAgent{2} begin
    is_susceptible::Bool
    is_infected::Bool
    is_recovered::Bool
end


space = GridSpace((50, 50); periodic=false)


function initialize(;
    number_S = 499,
    number_I = 1,
    number_R = 0,
    α = 0.1,  # used for the infection of susceptible individuals
    β = 0.01, # used for the resistance gained by the infectious individuals
    neighbours_size = 2, # Neighbour distance detection
    griddims = (50, 50),
    seed = 125
    )
    nb_hosts = number_S + number_I + number_R
    nb_infected = number_I
    
    # Initialize model parameters
    space = GridSpace(griddims; periodic=false)
    properties = @dict number_S number_I number_R nb_hosts α β neighbours_size    
    rng = Random.MersenneTwister(seed)
    scheduler = Schedulers.randomly
    
    model = ABM(Patient, space; properties, rng, scheduler)
    
    for i in 1:nb_hosts
        if i <= number_S
            patient = Patient(i, (1,1), true, false, false)
        elseif i <= number_S + number_I
            patient = Patient(i, (1,1), false, true, false)
        else 
            patient = Patient(i, (1,1), false, false, true)
        end
        add_agent_single!(patient, model)
    end
    
    return model
end
    
function basic_move!(agent, model)
    possible_directions = ( (1,0), (0,1), (-1,0), (0,-1) ) # Boundary case not handled
    walk!(agent, rand(model.rng, possible_directions), model)
end

function become_infected!(agent, model)
    if agent.is_susceptible
        count_infected_ngb = sum( [neighbor.is_infected for neighbor in nearby_agents(agent, model)] )
        probability_infect = min(1, model.α * count_infected_ngb)
        if isone(rand(model.rng, Binomial(1, probability_infect )))
            agent.is_susceptible = false
            agent.is_infected = true
        end
    end
end

function become_recoved!(agent, model)
    if agent.is_infected
        if isone(rand(model.rng, Binomial(1, model.β)))
            agent.is_infected = false
            agent.is_recovered = true
        end
    end
end
    
function agent_step!(agent, model)
    become_infected!(agent, model)
    become_recoved!(agent, model)
    basic_move!(agent, model)
end

function model_step!(model)
end

model = initialize()

## Visualizeation preparation
GLMakie.activate!() # hide

patient_color(a) = a.is_susceptible == 1 ? :green : (a.is_infected == 1 ? :red : :blue)
patient_shape(a) = :circle

# figure, _ = abm_plot(model; ac = patient_color, am = patient_shape, as = 10)
# figure # returning the figure displays it


## Interactive application
parange = Dict(:α => 0.01:0.5)
adata = [(:is_susceptible, sum), (:is_infected, sum), (:is_recovered, sum)]
alabels = ["S", "I", "R"]

figure, adf, mdf = abm_data_exploration(
    model, agent_step!, dummystep, parange;
    ac = patient_color, am = patient_shape, as = 10,
    adata, alabels)


## Plot simulation curve
using Plots, StatsPlots, DataFrames

model = initialize()
adata = [(:is_susceptible, sum), (:is_infected, sum), (:is_recovered, sum)]
mdata = [:number_S]
steps = 600
data, data2 = run!(model, agent_step!, steps; adata, mdata)


@df data Plots.plot(:step, [:sum_is_susceptible, :sum_is_infected, :sum_is_recovered])



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

Plots.plot(sir_solution,xlabel="Time",ylabel="Number")
