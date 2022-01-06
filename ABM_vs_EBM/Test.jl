using Agents

@agent SchellingAgent GridAgent{2} begin
    mood::Bool          # whether the agent is happy in its position. (true = happy)
    group::Int          # The group of the agent, determines mood as it interacts with neighbors    
end


using Random
function initialize(; numagents = 320, griddims = (20, 20), min_to_be_happy = 3, seed = 125)
    space = GridSpace(griddims; periodic=false)
    properties = Dict(:min_to_be_happy => min_to_be_happy)
    rng = Random.MersenneTwister(seed)
    
    model = ABM(
        SchellingAgent, space;
        properties, rng, scheduler = Schedulers.randomly
    )
    
    for n in 1:numagents
        agent = SchellingAgent(n, (1,1), false, n < numagents / 2 ? 1 : 2)
        add_agent_single!(agent, model)
    end
    return model
end


function agent_step!(agent, model)
    minhappy = model.min_to_be_happy
    count_neighbors_same_group = 0
    for neighbor in nearby_agents(agent, model)
        if agent.group == neighbor.group
            count_neighbors_same_group += 1
        end
    end
    
    if count_neighbors_same_group >= minhappy
        agent.mood = true
    else
        move_agent_single!(agent, model)
    end
    return
end


## Pass in f(agent)
model = initialize()
position(agent::SchellingAgent) = agent.pos[1]
adata = [position, :mood, :group]
data, _ = run!(model, agent_step!, 5; adata)
data[1:10, :] # print only a few rows


##
using Statistics: mean
model = initialize()
x(agent) = agent.pos[1]
adata = [(:mood, sum), (x, mean)]
data, _ = run!(model, agent_step!, 5; adata)

##
using InteractiveDynamics
using GLMakie # choosing a plotting backend
GLMakie.activate!() # hide

groupcolor(a) = a.group == 1 ? :blue : :orange
groupmarker(a) = a.group == 1 ? :circle : :rect
figure, _ = abm_plot(model; ac = groupcolor, am = groupmarker, as = 10)
figure # returning the figure displays it


## 
model = initialize();
abm_video(
    "schelling.mp4", model, agent_step!;
    ac = groupcolor, am = groupmarker, as = 10,
    framerate = 4, frames = 20,
    title = "Schelling's segregation model"
)

##
parange = Dict(:min_to_be_happy => 0:8)
adata = [(:mood, sum), (x, mean)]
alabels = ["happy", "avg. x"]
model = initialize(; numagents = 300)

figure, adf, mdf = abm_data_exploration(
    model, agent_step!, dummystep, parange;
    ac = groupcolor, am = groupmarker, as = 10,
    adata, alabels
)

figure