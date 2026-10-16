import numpy as np
import copy
import random
import bisect
from .ngram import predict_next_pitch, extract_valid_pitch_context
from .fitness.main import analyze_function, fitness_function, FitnessWeights, get_fitness_weights


class GA:
    def __init__(self,
                 num_generations,
                 population_size,
                 chromosome_length,
                 crossover_rate,
                 mutation_rate,
                 target_features,
                 origin_features,
                 tonic_offset=0,
                 mode='major',
                 hand='right',
                 k_tournament_size=3,
                 pitch_ngram_model=None,
                 ngram_parameter=3,
                 fitness_weights: FitnessWeights | None = None,
                 measure_boundaries=None,
                 ):
        self.num_generations = num_generations
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.target_features = target_features
        self.origin_features = origin_features
        self.tonic_offset = tonic_offset
        self.mode = mode
        self.hand = hand
        self.k_tournament_size = k_tournament_size
        self.pitch_ngram_model = pitch_ngram_model
        self.ngram_parameter = ngram_parameter
        self.fitness_weights = fitness_weights or get_fitness_weights("balanced")
        self.measure_boundaries = sorted({int(x) for x in (measure_boundaries or []) if 0 < int(x) < chromosome_length})


        self._fitness_cache = {}
        self._fitness_cache_order = []
        self._fitness_cache_cap = 5000

        self._empty_left_cache = {}
        self.origin_add = 0
        self._current_generation = 0

    def get_notes_positions(self, chromosome):
        """Return note-onset positions with and without rests."""
        positions_with_rest = []
        positions_without_rest = []
        for idx in range(len(chromosome)):
            current_element = chromosome[idx]
            if current_element != [-1] and current_element != [-3]:
                positions_with_rest.append(idx)
                if current_element:
                    positions_without_rest.append(idx)

        return (positions_with_rest, positions_without_rest)

    def generate_initial_population(self, reference):
        """Initialize the population by perturbing pitch, rhythm, and ornamentation."""
        self.population = [copy.deepcopy(reference)]

        for pop_idx in range(self.population_size-1):
            individual = copy.deepcopy(reference)
            positions_without_rest = self.get_notes_positions(individual)[1]
            num_notes = len(positions_without_rest)
            if num_notes == 0:
                self.population.append(individual)
                continue
            num_mutations = np.random.randint(0, max(2, num_notes // 5 + 1))
            notes_to_mutate = random.sample(positions_without_rest, num_mutations)

            for idx in notes_to_mutate:
                if(individual[idx] in ([], [-1], [-3])):
                    positions_without_rest = self.get_notes_positions(individual)[1]
                    replaced_idx = random.choice(positions_without_rest)
                    individual = self.mutation(individual, replaced_idx)
                else:
                    individual = self.mutation(individual, idx)

            self.population.append(individual)



    def mutation_entrance(self, individual):
        """Apply note-level mutations across eligible note onsets."""

        if self.mutation_rate == 0:
            return individual
        positions_without_rest = self.get_notes_positions(individual)[1]

        for abs_idx in positions_without_rest:
            if np.random.rand() < self.mutation_rate:
                individual = self.mutation(individual, abs_idx)
        return individual

    def mutation(self, individual, idx):
        """Apply note-level mutations across eligible note onsets."""
        element = individual[idx]
        is_chord = len(element) > 1



        mutation_type = random.choice(
            ['pitch', 'rhythm'] if is_chord else ['pitch', 'rhythm', 'Ornamentation']
        )

        if mutation_type == 'pitch':
            context = extract_valid_pitch_context(individual, idx, self.ngram_parameter)
            individual[idx] = self.pitch_Changing(element, context)

        elif mutation_type == 'rhythm':
            cur_positions = self.get_notes_positions(individual)[0]
            pos_i = bisect.bisect_left(cur_positions, idx)
            if pos_i < len(cur_positions) and cur_positions[pos_i] == idx:
                individual = self.Rhythm_Changing(individual, pos_i)

        elif mutation_type == 'Ornamentation' and len(individual[idx]) == 1:


            offset = self.Ornamentation_Changing(element)
            base = element[0] % 128
            individual[idx] = [base + offset]

        return individual

    def pitch_Changing(self, element, context):
        """Mutate pitch using the n-gram model when available, otherwise use a bounded semitone shift."""
        if(element in ([], [-1], [-3])):
            return element
        def clamp_base_pitch(pitch):
            val = pitch + self.tonic_offset
            return max(21, min(108, val)) - self.tonic_offset

        bases = [p % 128 for p in element]
        offsets = [p - (p % 128) if p >= 128 else 0 for p in element]

        if self.pitch_ngram_model and context:
            predicted = predict_next_pitch(self.pitch_ngram_model, context, self.ngram_parameter)
            if predicted is not None:
                root = bases[0]
                return [
                    clamp_base_pitch(predicted + (base - root)) + offset
                    for base, offset in zip(bases, offsets)
                ]


        shift = random.randint(-12, 12)
        return [clamp_base_pitch(base + shift) + offset for base, offset in zip(bases, offsets)]

    def Ornamentation_Changing(self, element):
        """Sample an ornament type and return its encoded pitch offset."""
        ORNAMENT_OFFSETS = {
            'acciaccatura': 128,
            'turn': 256,
            'inverted_turn': 384,
            'trill': 512,
            'upper_mordent': 640,
            'lower_mordent': 768,
        }

        ornament_type = random.choice(list(ORNAMENT_OFFSETS.keys()))
        offset = ORNAMENT_OFFSETS[ornament_type]
        return offset

    def Rhythm_Changing(self, individual, index):
        """Mutate the rhythm of the selected note by splitting or merging events."""
        positions = self.get_notes_positions(individual)[0]
        mutation_type = random.choice(['split', 'merge'])

        note_num = len(positions)
        chromosome_length = len(individual)
        if note_num == 0:
            return individual
        current_note_index = positions[index]
        current_note = individual[current_note_index]
        next_note_index = positions[index + 1] if index + 1 < note_num else None
        next_note = individual[next_note_index] if next_note_index is not None else None


        if mutation_type == 'split' and index + 1 < note_num and next_note_index - current_note_index > 1:
            split_point = (current_note_index + next_note_index) // 2
            individual[split_point] = current_note
        elif mutation_type == 'split' and index + 1 == note_num and chromosome_length - current_note_index > 1:
            split_point = (chromosome_length + current_note_index) // 2
            individual[split_point] = current_note


        elif mutation_type == 'merge' and index + 1 < note_num:
            individual[current_note_index] = random.choice([current_note, next_note])
            if(next_note_index + 1 >= chromosome_length):
                if(individual[next_note_index - 1] == [-3]):
                    individual[next_note_index] = [-3]
                else:
                    individual[next_note_index] = [-1]
            elif (
                ((current_note_index + 1 < chromosome_length) and (individual[current_note_index + 1] == [-3])) ^
                ((next_note_index + 1 < chromosome_length) and (individual[next_note_index + 1] == [-3]))
            ):
                random_type = random.choice([[-3], [-1]])
                N = chromosome_length

                if (individual[current_note_index + 1] == random_type
                    or (random_type == [-1] and current_note_index + 1 == next_note_index)):

                    s = next_note_index + 1
                    e = s
                    while e < N and individual[e] in ([-3], [-1]):
                        e += 1
                    if e > s:
                        individual[s:e] = [random_type] * (e - s)
                else:

                    s = current_note_index + 1
                    e = next_note_index
                    if e > s:
                        individual[s:e] = [random_type] * (e - s)
            else:
                if(individual[next_note_index - 1] == [-3]):
                    individual[next_note_index] = [-3]
                else:
                    individual[next_note_index] = [-1]

        return individual



    def tournament_selection(self, fitnesses):
        """Select the best individual from a random k-way tournament."""
        candidates = random.sample(range(len(fitnesses)), self.k_tournament_size)
        best = candidates[0]
        for c in candidates[1:]:
            if fitnesses[c] > fitnesses[best]:
                best = c
        return best



    def crossover(self, p1, p2):
        """Single-point crossover at a measure boundary."""
        limit = min(len(p1), len(p2))
        points = [b for b in self.measure_boundaries if 0 < b < limit]
        if not points:


            points = list(range(16, limit, 16))
        if not points:
            return copy.deepcopy(p1), copy.deepcopy(p2)
        point = random.choice(points)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]




    def _chrom_key(self, chrom):
        """Convert a chromosome and fitness phase into an immutable cache key."""

        phase = 1 if getattr(self, "_current_generation", 0) >= getattr(self, "origin_add", 0) else 0
        return (self.hand, phase, tuple(tuple(el) for el in chrom))

    def _fitness_cache_get(self, key):
        """Retrieve a cached fitness value and refresh its LRU position."""
        v = self._fitness_cache.get(key)
        if v is not None:

            try:
                self._fitness_cache_order.remove(key)
            except ValueError:
                pass
            self._fitness_cache_order.append(key)
        return v

    def _fitness_cache_put(self, key, value):
        """Store a fitness value while maintaining the bounded LRU cache."""
        if key in self._fitness_cache:
            self._fitness_cache[key] = value
            try:
                self._fitness_cache_order.remove(key)
            except ValueError:
                pass
            self._fitness_cache_order.append(key)
            return

        self._fitness_cache[key] = value
        self._fitness_cache_order.append(key)

        if len(self._fitness_cache_order) > self._fitness_cache_cap:
            old = self._fitness_cache_order.pop(0)
            self._fitness_cache.pop(old, None)

    def _get_empty_left(self, L):
        """Return a cached empty left-hand template of length ``L``."""
        tpl = self._empty_left_cache.get(L)
        if tpl is None:

            tpl = tuple([] for _ in range(L))
            self._empty_left_cache[L] = tpl
        return tpl
    def fitness(self, chromosome, gen=0):
        self._current_generation = gen
        key = self._chrom_key(chromosome)
        cached = self._fitness_cache_get(key)
        if cached is not None:
            return cached


        if self.hand == 'right':

            L = len(chromosome)
            empty_left = self._get_empty_left(L)
            input_data = {"right_hand": chromosome, "left_hand": empty_left}
        elif self.hand == 'left':
            input_data = {"right_hand": self.best_melody, "left_hand": chromosome}
        else:
            raise ValueError("Hand must be either 'right' or 'left'.")

        analyzed = analyze_function(input_data)
        score = fitness_function(
            data=analyzed,
            target=self.target_features,
            hand=self.hand,
            origin=self.origin_features,
            gen=gen,
            threshold=self.origin_add,
            weights=self.fitness_weights,
        )

        self._fitness_cache_put(key, score)
        return score


    def run(self, original_chromosome, best_melody=None, origin_add=0):
        self.generate_initial_population(original_chromosome)
        self.best_melody = best_melody
        self.origin_add = origin_add
        best_scores, avg_scores = [], []


        parent_fitnesses = [self.fitness(ind, 0) for ind in self.population]

        for gen in range(self.num_generations):


            if gen == self.origin_add and gen > 0:
                parent_fitnesses = [self.fitness(ind, gen) for ind in self.population]
            children = []


            while len(children) < self.population_size:
                i1 = self.tournament_selection(parent_fitnesses)
                i2 = self.tournament_selection(parent_fitnesses)
                p1 = self.population[i1]
                p2 = self.population[i2]

                if random.random() < self.crossover_rate:
                    c1, c2 = self.crossover(p1, p2)
                    c1 = copy.deepcopy(c1)
                    c2 = copy.deepcopy(c2)
                else:
                    c1 = copy.deepcopy(p1)
                    c2 = copy.deepcopy(p2)

                c1 = self.mutation_entrance(c1)
                c2 = self.mutation_entrance(c2)
                children.append(c1)
                children.append(c2)


            child_fitnesses = [self.fitness(ind, gen) for ind in children]


            all_pop = self.population + children
            all_fit = np.asarray(parent_fitnesses + child_fitnesses, dtype=np.float32)

            k = self.population_size
            idx_top = np.argpartition(all_fit, -k)[-k:]
            idx_top = idx_top[np.argsort(all_fit[idx_top])[::-1]]

            self.population = [all_pop[i] for i in idx_top]
            parent_fitnesses = [float(all_fit[i]) for i in idx_top]

            best_scores.append(parent_fitnesses[0])
            avg_scores.append(float(np.mean(parent_fitnesses)))

        best_idx = int(np.argmax(parent_fitnesses))
        best_chromosome = self.population[best_idx]
        return (best_chromosome, best_scores, avg_scores)

