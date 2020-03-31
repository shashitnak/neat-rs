#![allow(unused)]
#![deny(missing_docs)]

//! # neat-rs
//! 
//! Implementation of neat algorithm in rust

mod neat;
mod genotype;
mod traits;
mod random;
pub use neat::*;
pub use genotype::*;
pub use traits::*;

#[cfg(test)]
mod tests {
    use super::*;

    use gym::{GymClient, SpaceData::DISCRETE};

    fn calculate_cart(genome: &impl Gene, display: bool) -> f64 {
        let mut fitness = 0.0;
        
        let client = GymClient::default();
        let env = client.make("CartPole-v0");
        let mut input = [0f64; 4];

        let init = env.reset().unwrap().get_box().unwrap();
        for i in 0..4 {
            input[i] = init[i];
        }

        loop {
            let pred = genome.predict(&input)[0];

            let action = DISCRETE(if pred < 0. { 0 } else { 1 });
            let state = env.step(&action).unwrap();
            let input_box = state.observation.get_box().unwrap();
            if display {
                env.render();
            }
            for i in 0..4 {
                input[i] = input_box[i];
            }
            if state.is_done {
                break;
            }
            // println!("{}", state.reward);
            fitness += state.reward;
        }
        env.close();

        if display {
            // println!("Output = {:#?}", for_display);
            println!("Fitness = {}", fitness);
        }

        fitness
    }

    #[test]
    fn test_neat() {
        let mut neat = Neat::<Genotype>::new(4, 1, 100, 0.01);

        for i in 1..=100 {
            println!("---------Gen #{}--------", i);
            neat.next_generation(calculate_cart);
        }
    }
}