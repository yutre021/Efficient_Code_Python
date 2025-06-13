# Efficient_Code_Python
Writing code efficient in Python


### In English

Efficient Python code is defined by two primary factors:

* **Fast Runtime:** The code should execute as quickly as possible, minimizing its completion time.
* **Small Memory Footprint:** The code should consume a minimal amount of system resources, particularly RAM.

In essence, the goal is to write code that is both **fast** and **lightweight**.

---

### Em Português

Código Python eficiente é definido por dois fatores principais:

* **Tempo de Execução Rápido:** O código deve executar o mais rápido possível, minimizando seu tempo de conclusão.
* **Baixo Uso de Memória:** O código deve consumir uma quantidade mínima de recursos do sistema, especialmente memória RAM.

Em essência, o objetivo é escrever um código que seja **rápido** e **leve**.



### In English

#### The Python Standard Library

The Python Standard Library is a vast collection of modules, functions, and data types that are included with every standard Python installation. You don't need to install anything extra to use them.

It consists of three main components:

* **Built-in Types:** These are the fundamental data structures you use to store information, such as:
    * `list`, `tuple`, `set`, `dict`

* **Built-in Functions:** These are globally available functions that you can use at any time for common tasks, for example:
    * `print()`, `len()`, `range()`, `map()`, `zip()`

* **Built-in Modules:** These are separate files containing more specialized functions and tools that you can access by importing them. Examples include:
    * `os` (for operating system interactions)
    * `math` (for mathematical functions)
    * `collections` (for advanced data structures)
    * `itertools` (for creating complex iterators)

In short, the Standard Library provides a powerful and convenient foundation for building Python applications without needing external packages.

---

### Em Português

#### A Biblioteca Padrão do Python (The Python Standard Library)

A Biblioteca Padrão do Python é uma vasta coleção de módulos, funções e tipos de dados que são incluídos em toda instalação padrão do Python. Você não precisa instalar nada extra para usá-los.

Ela consiste em três componentes principais:

* **Tipos Embutidos (Built-in Types):** São as estruturas de dados fundamentais que você usa para armazenar informações, tais como:
    * `list`, `tuple`, `set`, `dict`

* **Funções Embutidas (Built-in Functions):** São funções globalmente disponíveis que você pode usar a qualquer momento para tarefas comuns, por exemplo:
    * `print()`, `len()`, `range()`, `map()`, `zip()`

* **Módulos Embutidos (Built-in Modules):** São arquivos separados que contêm funções e ferramentas mais especializadas, que você pode acessar importando-os. Exemplos incluem:
    * `os` (para interações com o sistema operacional)
    * `math` (para funções matemáticas)
    * `collections` (para estruturas de dados avançadas)
    * `itertools` (para criar iteradores complexos)

Em resumo, a Biblioteca Padrão fornece uma base poderosa e conveniente para construir aplicações em Python sem a necessidade de pacotes externos.


```python
# Código que é executado rapidamente para a tarefa em questão, minimiza o espaço de memória e segue os princípios de estilo de
# codificação do Python.
names = ['Jerry', 'Kramer', 'Elaine', 'George', 'Newman']

i = 0
new_list= []
while i < len(names):
    if len(names[i]) >= 6:
        new_list.append(names[i])
    i += 1
print(new_list)


better_list = []
for name in names:
    if len(name) >= 6:
        better_list.append(name)
print(better_list)

# Print the list created by using list comprehension
best_list = [name for name in names if len(name) >= 6]
print(best_list)

# Código Pythonico == código eficiente.

# Create a range object that goes from 0 to 5
nums = range(6)
print(type(nums))

# Convert nums to a list
nums_list = list(nums)
print(nums_list)

# Create a new list of odd numbers from 1 to 11 by unpacking a range object
nums_list2 = [*range(1, 12, 2)]
print(nums_list2)


# Rewrite the for loop to use enumerate
indexed_names = []
for i, name in enumerate(names):
    index_name = (i, name)
    indexed_names.append(index_name) 
print(indexed_names)

# Rewrite the above for loop using list comprehension
indexed_names_comp = [(i, name) for i, name in enumerate(names)]
print(indexed_names_comp)

# Unpack an enumerate object with a starting index of one
indexed_names_unpack = [*enumerate(names, 1)]
print(indexed_names_unpack)


# Use map to apply str.upper to each element in names
names_map  = map(str.upper, names)

# Print the type of the names_map
print(type(names_map))

# Unpack names_map into a list
names_uppercase = [*names_map]

# Print the list created above
print(names_uppercase)


#Um array bidimensional numpy foi carregado na sua sessão (chamado nums) e impresso no console para sua #conveniência.
#numpy foi importado para a sua sessão como np.
# Print second row of nums
print(nums[1,:])

# Print all elements of nums that are greater than six
print(nums[nums > 6])

# Double every element of nums
nums_dbl = nums * 2
print(nums_dbl)

# Replace the third column of nums
nums[:,2] = nums[:,2] + 1
print(nums)


# Create a list of arrival times
arrival_times = [*range(10, 60, 10)]

print(arrival_times)


# Create a list of arrival times
arrival_times = [*range(10,60,10)]

# Convert arrival_times to an array and update the times
arrival_times_np = np.array(arrival_times)
new_times = arrival_times_np - 3

print(new_times)



# Create a list of arrival times
arrival_times = [*range(10,60,10)]

# Convert arrival_times to an array and update the times
arrival_times_np = np.array(arrival_times)
new_times = arrival_times_np - 3

# Use list comprehension and enumerate to pair guests to new times
guest_arrivals = [(names[i], time) for i, time in enumerate(new_times)]

print(guest_arrivals)



# Create a list of arrival times
arrival_times = [*range(10,60,10)]

# Convert arrival_times to an array and update the times
arrival_times_np = np.array(arrival_times)
new_times = arrival_times_np - 3

# Use list comprehension and enumerate to pair guests to new times
guest_arrivals = [(names[i],time) for i,time in enumerate(new_times)]

# Map the welcome_guest function to each (guest,time) pair
welcome_map = map(welcome_guest, guest_arrivals)

guest_welcomes = [*welcome_map]
print(*guest_welcomes, sep='\n')


#Examining runtime

# Create a list of integers (0-50) using list comprehension
%timeit nums_list_comp = [num for num in range(51)]
print(nums_list_comp)

# Create a list of integers (0-50) by unpacking range
%timeit nums_unpack = [*range(51)]
print(nums_unpack)

#%timeit -r5 -n25 set(heroes)


# Create a list using the formal name
formal_list = list()
print(formal_list)

# Create a list using the literal syntax
literal_list = []
print(literal_list)

# Create a list using the formal name
formal_list = list()
print(formal_list)

# Create a list using the literal syntax
literal_list = []
print(literal_list)

# Print out the type of formal_list
print(type(formal_list))

# Print out the type of literal_list
print(type(literal_list))

# Não inclua as declarações do site print() quando você estiver fazendo o timing.


literal_list = []
literal_dict = {}
literal_tuple = ()

formal_list = list()
formal_dict = dict()
formal_tuple = tuple()

l_time = %timeit -o literal_dict = {}

l_time = %timeit -o literal_dict = {}

diff = (f_time.average - l_time.average) * (10**9)
print('l_time better than f_time by {} ns'.format(diff))


# Comparing times
%timeit formal_dict = dict()

%timeit formal_dict = dict()


# pip install line_profiler


heroes = ['Batman', 'Superman', 'Wonder Woman']
hts = np.array([188.0, 191.0, 183.0])
wts = np.array([ 95.0, 101.0, 74.0])

# WRITING EFFICIENT PYTHON CODE
def convert_units(heroes, heights, weights):
    new_hts = [ht * 0.39370 for ht in heights]
    new_wts = [wt * 2.20462 for wt in weights]
    hero_data = {}
    for i,hero in enumerate(heroes):
        hero_data[hero] = (new_hts[i], new_wts[i])
    return hero_data


convert_units(heroes, hts, wts)


%timeit convert_units(heroes, hts, wts)

%timeit new_hts = [ht * 0.39370 for ht in hts]

%timeit new_wts = [wt * 2.20462 for wt in wts]

%%timeit
hero_data = {}
for i,hero in enumerate(heroes):
    hero_data[hero] = (new_hts[i], new_wts[i])

# In console

%load_ext line_profiler

%lprun -f convert_units convert_units(heroes, hts, wts)


%timeit convert_units convert_units(heroes, hts, wts)

#Quick and dirty approach
import sys

nums_list = [*range(1000)]
sys.getsizeof(nums_list)

# Code profiling: memory

# pip install memory_profiler

%load_ext memory_profiler
%mprun -f convert_units convert_units(heroes, hts, wts)

%load_ext memory_profiler
%mprun -f convert_units convert_units(heroes, hts, wts)

# Use get_publisher_heroes() to gather Star Wars heroes
star_wars_heroes = get_publisher_heroes(heroes, publishers, 'George Lucas')
print(star_wars_heroes)
print(type(star_wars_heroes))

# Use get_publisher_heroes_np() to gather Star Wars heroes
star_wars_heroes_np = get_publisher_heroes_np(heroes, publishers, 'George Lucas')

print(star_wars_heroes_np)
print(type(star_wars_heroes_np))

# Combining objects

names = ['Bulbasaur', 'Charmander', 'Squirtle']
hps = [45, 39, 44]

combined = []
for i,pokemon in enumerate(names):
    combined.append((pokemon, hps[i]))
print(combined)


combined_zip = zip(names, hps)
print(type(combined_zip))

combined_zip_list = [*combined_zip]
print(combined_zip_list)



# Counting with loop

# Each Pokémon's type (720 total)
poke_types = ['Grass', 'Dark', 'Fire', 'Fire', ...]
type_counts = {}
for poke_type in poke_types:
if poke_type not in type_counts:
type_counts[poke_type] = 1
else:
type_counts[poke_type] += 1
print(type_counts)


#collections.Counter()

# Each Pokémon's type (720 total)
poke_types = ['Grass', 'Dark', 'Fire', 'Fire', ...]
from collections import Counter
type_counts = Counter(poke_types)
print(type_counts)


poke_types = ['Bug', 'Fire', 'Ghost', 'Grass', 'Water']
combos = []
for x in poke_types:
    for y in poke_types:
        if x == y:
            continue
        if ((x,y) not in combos) & ((y,x) not in combos):
            combos.append((x,y))
print(combos)

# itertools.combinations()

poke_types = ['Bug', 'Fire', 'Ghost', 'Grass', 'Water']
from itertools import combinations
combos_obj = combinations(poke_types, 2)
print(type(combos_obj))


# Combine names and primary_types
names_type1 = [*zip(names, primary_types)]

print(*names_type1[:5], sep='\n')

# Combine all three lists together
names_types = [*zip(names, primary_types, secondary_types)]

print(*names_types[:5], sep='\n')


# Combine five items from names and three items from primary_types
differing_lengths = [*zip(names[:5], primary_types[:3])]

print(*differing_lengths, sep='\n')

# Combine five items from names and three items from primary_types
differing_lengths = [*zip(names[:5], primary_types[:3])]

print(*differing_lengths, sep='\n')


# Combine all three lists together
names_types = [*zip(names, primary_types, secondary_types)]

print(*names_types[:5], sep='\n')




# Collect the count of primary types
type_count = Counter(primary_types)
print(type_count, '\n')

# Collect the count of generations
gen_count = Counter(generations)
print(gen_count, '\n')

# Use list comprehension to get each Pokémon's starting letter
starting_letters = [name[0] for name in names]

# Collect the count of Pokémon for each starting_letter
starting_letters_count = Counter(starting_letters)
print(starting_letters_count)

# Import combinations from itertools
from itertools import combinations

# Create a combination object with pairs of Pokémon
combos_obj = combinations(pokemon, 2)
print(type(combos_obj), '\n')

# Convert combos_obj to a list by unpacking
combos_2 = [*combos_obj]
print(combos_2, '\n')

# Collect all possible combinations of 4 Pokémon directly into a list
combos_4 = [*combinations(pokemon, 4)]
print(combos_4)

# Convert both lists to sets
ash_set = set(ash_pokedex)
misty_set = set(misty_pokedex)

# Find the Pokémon that exist in both sets
both = ash_set.intersection(misty_set)
print(both)

# Find the Pokémon that Ash has and Misty does not have
ash_only = ash_set.difference(misty_set)
print(ash_only)

# Find the Pokémon that are in only one set (not both)
unique_to_set = ash_set.symmetric_difference(misty_set)
print(unique_to_set)

# Convert Brock's Pokédex to a set
brock_pokedex_set = set(brock_pokedex)
print(brock_pokedex_set)

# Check if Psyduck is in Ash's list and Brock's set
print('Psyduck' in ash_pokedex)
print('Psyduck' in brock_pokedex_set)

# Check if Machop is in Ash's list and Brock's set
print('Machop' in ash_pokedex)
print('Machop' in brock_pokedex_set)

# Use find_unique_items() to collect unique Pokémon names
uniq_names_func = find_unique_items(names)
print(len(uniq_names_func))

# Convert the names list to a set to collect unique Pokémon names
uniq_names_set = set(names)
print(len(uniq_names_set))

# Check that both unique collections are equivalent
print(sorted(uniq_names_func) == sorted(uniq_names_set))

# Use the best approach to collect unique primary types and generations
uniq_types = set(primary_types)
uniq_gens = set(generations)
print(uniq_types, uniq_gens, sep='\n') 


#Using holistic conversions
names = ['Pikachu', 'Squirtle', 'Articuno', ...]
legend_status = [False, False, True, ...]
generations = [1, 1, 1, ...]
poke_data_tuples = []
for poke_tuple in zip(names, legend_status, generations):
    poke_data_tuples.append(poke_tuple)
poke_data = [*map(list, poke_data_tuples)]
print(poke_data)


# Eliminate loops with NumPy
avgs = []
for row in poke_stats:
    avg = np.mean(row)
    avgs.append(avg)
print(avgs)


# Collect Pokémon that belong to generation 1 or generation 2
gen1_gen2_pokemon = [name for name,gen in zip(poke_names, poke_gens) if gen < 3]

# Create a map object that stores the name lengths
name_lengths_map = map(len, gen1_gen2_pokemon)

# Combine gen1_gen2_pokemon and name_lengths_map into a list
gen1_gen2_name_lengths = [*zip(gen1_gen2_pokemon, name_lengths_map)]

print(gen1_gen2_name_lengths_loop[:5])
print(gen1_gen2_name_lengths[:5])


# Create a total stats array
total_stats_np = stats.sum(axis=1)

# Create an average stats array
avg_stats_np = stats.mean(axis=1)

# Combine names, total_stats_np, and avg_stats_np into a list
poke_list_np = [*zip(names, total_stats_np, avg_stats_np)]

print(poke_list_np == poke_list, '\n')
print(poke_list_np[:3])
print(poke_list[:3], '\n')
top_3 = sorted(poke_list_np, key=lambda x: x[1], reverse=True)[:3]
print('3 strongest Pokémon:\n{}'.format(top_3))


# Import Counter
from collections import Counter

# Collect the count of each generation
gen_counts = Counter(generations)

# Improve for loop by moving one calculation above the loop
total_count = len(generations)
for gen,count in gen_counts.items():
    gen_percent = round(count / total_count * 100, 2)
    print('generation {}: count = {:3} percentage = {}'
          .format(gen, count, gen_percent))

# Collect all possible pairs using combinations()
possible_pairs = [*combinations(pokemon_types, 2)]

# Create an empty list called enumerated_tuples
enumerated_tuples = []

for i,pair in enumerate(possible_pairs, 1):
    enumerated_pair_tuple = (i,) + pair
    enumerated_tuples.append(enumerated_pair_tuple)

# Convert all tuples in enumerated_tuples to a list
enumerated_pairs = [*map(list, enumerated_tuples)]
print(enumerated_pairs)


# Calculate the total HP avg and total HP standard deviation
hp_avg = hps.mean()
hp_std = hps.std()

# Use NumPy to eliminate the previous for loop
z_scores = (hps - hp_avg)/hp_std

# Combine names, hps, and z_scores
poke_zscores2 = [*zip(names, hps, z_scores)]
print(*poke_zscores2[:3], sep='\n')

hp_avg = hps.mean()
hp_std = hps.std()

# Use NumPy to eliminate the previous for loop
z_scores = (hps - hp_avg)/hp_std

# Combine names, hps, and z_scores
poke_zscores2 = [*zip(names, hps, z_scores)]
print(*poke_zscores2[:3], sep='\n')

# Use list comprehension with the same logic as the highest_hp_pokemon code block
highest_hp_pokemon2 = [(name, hp, zscore) for name, hp, zscore in poke_zscores2 if zscore > 2]
print(*highest_hp_pokemon2, sep='\n')


# Iterating with .iloc

%%timeit
win_perc_list = []
for i in range(len(baseball_df)):
    row = baseball_df.iloc[i]
    wins = row['W']
    games_played = row['G']
    win_perc = calc_win_perc(wins, games_played)
    win_perc_list.append(win_perc)

baseball_df['WP'] = win_perc_list

# Iterating with .iterrows()

win_perc_list = []
for i,row in baseball_df.iterrows():
    wins = row['W']
    games_played = row['G']
    win_perc = calc_win_perc(wins, games_played)
    win_perc_list.append(win_perc)

baseball_df['WP'] = win_perc_list


# Iterating with .itertuples()

for row_namedtuple in team_wins_df.itertuples():
    print(row_namedtuple)

# Iterate over pit_df and print each row
for i, row in pit_df.iterrows():
    print(row)

# Iterate over pit_df and print each index variable, row, and row type
for i,row in pit_df.iterrows():
    print(i)
    print(row)
    print(type(row))


# Use one variable instead of two to store the result of .iterrows()
for row_tuple in pit_df.iterrows():
    print(row_tuple)

# Print the row and type of each row
for row_tuple in pit_df.iterrows():
    print(row_tuple)
    print(type(row_tuple))


# Create an empty list to store run differentials
run_diffs = []

# Write a for loop and collect runs allowed and runs scored for each row
for i,row in giants_df.iterrows():
    runs_scored = row['RS']
    runs_allowed = row['RA']
    
    # Use the provided function to calculate run_diff for each row
    run_diff = calc_run_diff(runs_scored, runs_allowed)
    
    # Append each run differential to the output list
    run_diffs.append(run_diff)

giants_df['RD'] = run_diffs
print(giants_df)


# Loop over the DataFrame and print each row
for row in rangers_df.itertuples():
  print(row)

# Loop over the DataFrame and print each row's Index, Year and Wins (W)
for row in rangers_df.itertuples():
  i = row.Index
  year = row.Year
  wins = row.W
  print(i, year, wins)

run_diffs = []

# Loop over the DataFrame and calculate each row's run differential
for row in yankees_df.itertuples():
    
    runs_scored = row.RS
    runs_allowed = row.RA

    run_diff = calc_run_diff(runs_scored, runs_allowed)
    
    run_diffs.append(run_diff)

# Append new column
yankees_df['RD'] = run_diffs
print(yankees_df)


# Gather sum of all columns
stat_totals = rays_df.apply(sum, axis=0)
print(stat_totals)




# Display the first five rows of the DataFrame
print(dbacks_df.head())

# Create a win percentage Series 
win_percs = dbacks_df.apply(lambda row: calc_win_perc(row['W'], row['G']), axis=1)
print(win_percs, '\n')

# Append a new column to dbacks_df
dbacks_df['WP'] = win_percs
print(dbacks_df, '\n')

# Display dbacks_df where WP is greater than 0.50
print(dbacks_df[dbacks_df['WP'] >= 0.50])


# Use the W array and G array to calculate win percentages
win_percs_np = calc_win_perc(baseball_df['W'].values, baseball_df['G'].values)

# Append a new column to baseball_df that stores all win percentages
baseball_df['WP'] = win_percs_np

print(baseball_df.head())

win_perc_preds_loop = []

# Use a loop and .itertuples() to collect each row's predicted win percentage
for row in baseball_df.itertuples():
    runs_scored = row.RS
    runs_allowed = row.RA
    win_perc_pred = predict_win_perc(runs_scored, runs_allowed)
    win_perc_preds_loop.append(win_perc_pred)

# Apply predict_win_perc to each row of the DataFrame
win_perc_preds_apply = baseball_df.apply(lambda row: predict_win_perc(row['RS'], row['RA']), axis=1)

# Calculate the win percentage predictions using NumPy arrays
win_perc_preds_np = predict_win_perc(baseball_df['RS'].values, baseball_df['RA'].values)
baseball_df['WP_preds'] = win_perc_preds_np
print(baseball_df.head())
```
