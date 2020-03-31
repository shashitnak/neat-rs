
/// Trait required for a genome to be used by neat
pub trait Gene {
    /// returns an empty genome with only input and output nodes and no connections
    fn empty(inputs: usize, outputs: usize) -> Self;
    /// checks if the other genome is same species as self
    fn is_same_species_as(&self, other: &Self) -> bool;
    /// method for cross over of two genomes
    fn cross(&self, other: &Self) -> Self;
    /// method for mutation of the genome
    fn mutate<T: GlobalNeatCounter>(self, neat: &mut T) -> Self;
    /// constructs the neural network and returns the output as a vec of floats
    fn predict(&self, input: &[f64]) -> Vec<f64>;
}


/// Trait required by neat struct
pub trait GlobalNeatCounter {
    /// returns the innovation number of new connection if a connection can be added, otherwise returns None
    fn try_adding_connection(&mut self, from: usize, to: usize) -> Option<usize>;
    /// returns the index of new node
    fn get_new_node(&mut self) -> usize;
}