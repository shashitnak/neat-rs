
use super::traits::*;
use std::collections::HashSet;
use rand::random;

fn randint(n: usize) -> usize {
    ((n as f64)*random::<f64>()) as usize
}

fn pick_one(list: &[f64]) -> usize {
    let total: f64 = list.iter().sum();
    let mut r = random::<f64>();

    for (index, &item) in list.iter().enumerate() {
        let prob = item / total;
        r -= prob;
        if (r < 0.) {
            return index;
        }
    }

    list.len() - 1
}

/// Neat struct that takes care of evolving the population based on the fitness scores
#[derive(Debug)]
pub struct Neat<T: Gene> {
    nodes: usize,
    connections: HashSet<(usize, usize)>,
    genomes: Vec<T>,
    mutation_rate: f64
}

impl<T: Gene> Neat<T> {
    /// Creates a new population
    pub fn new(inputs: usize, outputs: usize, size: usize, mutation_rate: f64) -> Self {
        Self {
            nodes: 1 + inputs + outputs,
            connections: HashSet::new(),
            genomes: (0..size).map(|_| Gene::empty(inputs, outputs)).collect(),
            mutation_rate
        }
    }

    fn speciate(&self) -> Vec<Vec<usize>> {
        let mut speciess = Vec::new();
        speciess.push(vec![0]);

        for i in 1..self.genomes.len() {
            let mut add_to = 0;
            for species in &mut speciess {
                let index = species[randint(species.len())];
                if self.genomes[i].is_same_species_as(&self.genomes[index]) {
                    species.push(i);
                    break;
                }
                add_to += 1;
            }
            if add_to >= speciess.len() {
                speciess.push(vec![i]);
            }
        }
        speciess
    }

    fn calculate_fitness(&self, calculate: fn(&T, bool) -> f64) -> (Vec<f64>, f64) {
        let mut total_score = 0.;
        let mut best_score = 0.;
        let mut fittest = &self.genomes[0];
        let mut scores = Vec::new();

        for genome in &self.genomes {
            let score = calculate(genome, false);
            if score > best_score {
                best_score = score;
                fittest = genome;
            }
            scores.push(score);
            total_score += score;
        }
        calculate(fittest, true);
        (scores, total_score)
    }

    /// Takes care of evaluation, speciation, selection and mutation and creates the
    /// new population
    pub fn next_generation(&mut self, calculate: fn(&T, bool) -> f64) {
        let (scores, total_score) = self.calculate_fitness(calculate);

        let total_mean = total_score / (scores.len() as f64);

        let mut speciess = self.speciate();

        println!("Number of species = {}", speciess.len());
        println!("Number of genomes = {}", self.genomes.len());

        let mut next_generation = Vec::new();

        for species in &mut speciess {
            // Sort species based on fitness
            species.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());

            let top = (0.6 * species.len() as f64).round() as usize;

            let species_scores: Vec<_> = species
                .iter()
                .map(|&i| scores[i])
                .collect();

            let num: f64 = species_scores.iter().sum();
            let num = (num / total_mean).round() as usize;

            for _ in 0..num {
                let index1 = pick_one(&species_scores[..top]);
                let index2 = pick_one(&species_scores[..top]);

                let one = &self.genomes[index1];
                let two = &self.genomes[index2];

                let mut child = if scores[index1] > scores[index2] {
                    one.cross(two)
                } else {
                    two.cross(one)
                };

                if random::<f64>() < self.mutation_rate {
                    child = child.mutate(self);
                }

                next_generation.push(child);
            }
        }

        self.genomes = next_generation;
    }
}

impl<T: Gene> GlobalNeatCounter for Neat<T> {
    fn try_adding_connection(&mut self, from: usize, to: usize) -> Option<usize> {
        let innov_num = self.connections.len();

        if self.connections.insert((from, to)) {
            Some(innov_num)
        } else {
            None
        }
    }

    fn get_new_node(&mut self) -> usize {
        let new_node = self.nodes;
        self.nodes += 1;
        new_node
    }
}
