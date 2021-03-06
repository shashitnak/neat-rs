
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
        let mut species_plural = Vec::new();
        species_plural.push(vec![0]);

        for i in 1..self.genomes.len() {
            let mut added = false;
            for species in &mut species_plural {
                let index = species[randint(species.len())];
                if self.genomes[i].is_same_species_as(&self.genomes[index]) {
                    species.push(i);
                    added = true;
                    break;
                }
            }
            if !added {
                species_plural.push(vec![i]);
            }
        }
        species_plural
    }

    /// Evaluates all the genomes based on the fitness function and returns all the scores
    /// and the total score
    pub fn calculate_fitness(&self, calculate: impl Fn(&T, bool) -> f64) -> (Vec<f64>, f64) {
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

    /// Takes care of speciation, selection and mutation and creates the
    /// new population
    pub fn next_generation(&mut self, scores: &[f64], total_score: f64) {
        let total_mean = total_score / (scores.len() as f64);

        let mut species_plural = self.speciate();

        println!("Number of species = {}", species_plural.len());
        println!("Number of genomes = {}", self.genomes.len());

        let mut next_generation = Vec::new();

        for species in &mut species_plural {
            // Sort species based on fitness
            species.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());

            let top = (0.6 * species.len() as f64).round() as usize;

            let num: f64 = species
                .iter()
                .map(|&i| scores[i])
                .sum();
            
            let num = (num / total_mean).round() as usize;

            let species_scores: Vec<_> = species
            .iter()
            .take(top)
            .map(|&i| scores[i])
            .collect();

            for _ in 0..num {
                let index1 = species[pick_one(&species_scores)];
                let index2 = species[pick_one(&species_scores)];

                let one = &self.genomes[index1];
                let two = &self.genomes[index2];

                let mut child = if scores[index1] > scores[index2] {
                    one.cross(two)
                } else {
                    two.cross(one)
                };

                if random::<f64>() < self.mutation_rate {
                    child.mutate(self);
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_po() {
        let scores = [100., 1., 2., 4., 5., 8., 92.];
        for _ in 0..100 {
            for _ in 0..50 {
                print!("{} ", pick_one(&scores));
            }
        }
        println!();
    }
}