@agent Patient GridAgent{2} begin
    is_susceptible::Bool
    is_infected::Bool
    is_recovered::Bool
    is_collision::Bool
end

function initialize(;
    number_S = 499,
    number_I = 1,
    number_R = 0,
    α = 0.05,  # used for the infection of susceptible individuals
    β = 0.01, # used for the resistance gained by the infectious individuals
    neighbours_size = 2, # Neighbour distance detection
    griddims = (50, 50),
    meet_prob = 1.,
    rng = Xoroshiro128Plus()
    )
    nb_hosts = number_S + number_I + number_R
    nb_infected = number_I
    collision_ratio = .0
    
    # Initialize model parameters
    space = GridSpace(griddims; periodic=false)
    properties = @dict number_S number_I number_R nb_hosts α β neighbours_size meet_prob collision_ratio 
    # seed = 125
    # rng = Random.MersenneTwister(seed)
    scheduler = Schedulers.randomly
    
    model = ABM(Patient, space; properties, rng, scheduler)
    
    for i in 1:nb_hosts
        if i <= number_S
            patient = Patient(i, (1,1), true, false, false, false)
        elseif i <= number_S + number_I
            patient = Patient(i, (1,1), false, true, false, false)
        else 
            patient = Patient(i, (1,1), false, false, true, false)
        end
        add_agent_single!(patient, model)
    end
    
    return model
end

function become_infected!(agent, model)
    if agent.is_susceptible
        count_infected_ngb = sum( [neighbor.is_infected for neighbor in nearby_agents(agent, model)] )
        probability_infect = min(1, (model.α  * model.meet_prob) * count_infected_ngb) # Critical step
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

function basic_move!(agent, model)
    possible_directions = ( (1,0), (0,1), (-1,0), (0,-1) ) # Boundary case not handled
    walk!(agent, rand(model.rng, possible_directions), model)
end

function detect_collision!(agent, model)
    count_ngb = length([neighbor.id for neighbor in nearby_agents(agent, model)])
    if !iszero(count_ngb)
        agent.is_collision = true
    end
end
    
function agent_step!(agent, model)
    become_infected!(agent, model)
    become_recoved!(agent, model)
    detect_collision!(agent, model)
    basic_move!(agent, model)
end

## Model

function reset_collison_ratio!(model)
    model.collision_ratio = sum( [a.is_collision for a in allagents(model)] ) / model.nb_hosts
    for a in allagents(model)
        a.is_collision = false
    end
end

function model_step!(model)
    reset_collison_ratio!(model)
end