import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from deap import base, creator, tools, algorithms

# Define fuzzy variables and membership functions
arrival_last_month = ctrl.Antecedent(np.arange(0, 101, 1), 'arrival_last_month')
arrival_two_months_ago = ctrl.Antecedent(np.arange(0, 101, 1), 'arrival_two_months_ago')
predicted_arrival = ctrl.Consequent(np.arange(0, 101, 1), 'predicted_arrival')

# Triangular membership functions
arrival_last_month['low'] = fuzz.trimf(arrival_last_month.universe, [0, 0, 50])
arrival_last_month['medium'] = fuzz.trimf(arrival_last_month.universe, [0, 50, 100])
arrival_last_month['high'] = fuzz.trimf(arrival_last_month.universe, [50, 100, 100])

arrival_two_months_ago['low'] = fuzz.trimf(arrival_two_months_ago.universe, [0, 0, 50])
arrival_two_months_ago['medium'] = fuzz.trimf(arrival_two_months_ago.universe, [0, 50, 100])
arrival_two_months_ago['high'] = fuzz.trimf(arrival_two_months_ago.universe, [50, 100, 100])

predicted_arrival['low'] = fuzz.trimf(predicted_arrival.universe, [0, 0, 50])
predicted_arrival['medium'] = fuzz.trimf(predicted_arrival.universe, [0, 50, 100])
predicted_arrival['high'] = fuzz.trimf(predicted_arrival.universe, [50, 100, 100])

# Step 2: Define fuzzy rule base
rule1 = ctrl.Rule(arrival_last_month['low'] & arrival_two_months_ago['low'], predicted_arrival['low'])
rule2 = ctrl.Rule(arrival_last_month['medium'] & arrival_two_months_ago['medium'], predicted_arrival['medium'])
rule3 = ctrl.Rule(arrival_last_month['high'] & arrival_two_months_ago['high'], predicted_arrival['high'])
rule4 = ctrl.Rule(arrival_last_month['low'] & arrival_two_months_ago['high'], predicted_arrival['medium'])
rule5 = ctrl.Rule(arrival_last_month['high'] & arrival_two_months_ago['low'], predicted_arrival['medium'])

# Create fuzzy control system
fuzzy_control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
fuzzy_simulation = ctrl.ControlSystemSimulation(fuzzy_control_system)

# Function for prediction
def forecast(arrival_last, arrival_two_months):
    fuzzy_simulation.input['arrival_last_month'] = arrival_last
    fuzzy_simulation.input['arrival_two_months_ago'] = arrival_two_months
    fuzzy_simulation.compute()
    return fuzzy_simulation.output['predicted_arrival']

# Step 3: Optimize rules using genetic algorithm

# Define fitness function for evaluating prediction accuracy
def fitness(individual):
    # Use sample data for fitness evaluation
    sample_data = [(20, 30, 25), (70, 80, 75), (50, 50, 50)]
    mse = 0
    for arrival_last, arrival_two_months, actual in sample_data:
        prediction = forecast(arrival_last, arrival_two_months)
        mse += (prediction - actual) ** 2
    return (mse / len(sample_data)),

# Genetic algorithm settings
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.rand)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Create initial population and run genetic algorithm
population = toolbox.population(n=10)
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)

# Display best result
best_individual = tools.selBest(population, k=1)[0]
print("Best individual:", best_individual)
