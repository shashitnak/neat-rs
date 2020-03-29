#![allow(unused)]

//! # neat-rs
//! 
//! Implementation of neat algorithm in rust

pub use neat::*;
extern crate rand;

mod neat {

    use rand::random;

    fn randint(n: usize) -> usize {
        ((n as f64)*random::<f64>()) as usize
    }

    fn randf(n: f64) -> f64 {
        n*(2.*random::<f64>() - 1.)
    }

    fn random_weight() -> f64 {
        randf(2.)
    }

    fn random_bias() -> f64 {
        randf(30.)
    }

    #[derive(Debug, Clone)]
    enum Node {
        Input,
        Bias,
        Hidden(u128),
        Output
    }

    use Node::*;

    impl Node {
        fn value(&self) -> u128 {
            match *self {
                Input | Bias => 0,
                Hidden(val) => val,
                Output => std::u128::MAX
            }
        }

        fn can_connect_to(&self, to: &Self) -> bool {
            self.value() < to.value()
        }

        fn new_hidden(from: &Self, to: &Self) -> Option<Self> {
            match (from.value(), to.value()) {
                (val1, val2) if val1 <= val2 => Some(Hidden(val1 + (val2 - val1) / 2)),
                _ => None
            }
        }
    }

    #[derive(Debug, Clone)]
    enum ConnectionState {
        Disabled,
        Enabled
    }

    use ConnectionState::*;

    impl ConnectionState {
        fn toggle(&mut self) {
            *self = match *self {
                Enabled => Disabled,
                Disabled => Enabled
            };
        }

        fn enable(&mut self) {
            *self = Enabled;
        }

        fn disable(&mut self) {
            *self = Disabled;
        }
    }

    /// Connection gene provided by the Genome struct that can be used to build the neural network
    #[derive(Debug, Clone)]
    pub struct Connection {
        pub from: usize,
        pub to: usize,
        pub weight: f64,
        state: ConnectionState
    }

    impl Connection {
        fn new(from: usize, to: usize, weight: f64) -> Self {
            Self {
                from,
                to,
                weight,
                state: Enabled
            }
        }

        fn disable(&mut self) {
            self.state.disable();
        }

        fn enable(&mut self) {
            self.state.enable();
        }

        fn toggle(&mut self) {
            self.state.toggle();
        }

        fn shift_weight(&mut self) {
            self.weight *= 0.95;
        }

        fn change_weight(&mut self, weight: f64) {
            self.weight = weight;
        }
        /// Checks if the connection is enabled of not
        pub fn is_enabled(&self) -> bool {
            match self.state {
                Enabled => true,
                Disabled => false
            }
        }
    }

    /// Genotype representation of the network
    #[derive(Debug)]
    pub struct Genome {
        nodes: Vec<Option<Node>>,
        pub conns: Vec<Option<Connection>>,
        pub bias: f64,
        bio: usize
    }

    impl Genome {
        fn empty(inputs: usize, outputs: usize) -> Self {
            let nodes = (0..1).map(|_| Some(Bias))
                .chain((0..inputs).map(|_| Some(Input)))
                .chain((0..outputs).map(|_| Some(Output)))
                .collect();
            Self {
                nodes,
                conns: Vec::new(),
                bias: random_bias(),
                bio: 1 + inputs + outputs
            }
        }

        pub fn hidden(&self) -> usize {
            self.nodes.len() - self.bio
        }

        fn distance_from(&self, other: &Self) -> f64 {
            let mut disjoint_genes = 0.;
            let mut delta_w = 0.;

            let len1 = self.conns.len();
            let len2 = other.conns.len();

            let (excess_genes, len) = if len1 > len2 {
                (len1 - len2, len2)
            } else if len2 > len1 {
                (len2 - len1, len1)
            } else {
                (0, len1)
            };

            let excess_genes = excess_genes as f64;

            let mut n1: f64 = 0.;
            let mut n2: f64 = 0.;

            for i in 0..len {
                match (&self.conns[i], &other.conns[i]) {
                    (Some(conns1), Some(conns2)) => {
                        delta_w += (conns1.weight - conns2.weight).abs();
                        n1 += 1.;
                        n2 += 1.;
                    },
                    (Some(_), None) => {
                        disjoint_genes += 1.;
                        n1 += 1.;
                    },
                    (None, Some(_)) => {
                        disjoint_genes += 1.;
                        n2 += 1.;
                    },
                    _ => {}
                }
            }

            let mut n = n1.max(n2);

            if n < 20. {
                n = 1.;
            }

            (excess_genes / n) + (disjoint_genes / n) + 0.5*delta_w
        }

        fn is_same_species_as(&self, other: &Self) -> bool {
            self.distance_from(other) < 2.
        }

        fn cross(&self, other: &Self) -> Self {
            let mut nodes: Vec<_> = self
                .nodes
                .iter()
                .take_while(|x| match x {
                    Some(Hidden(_)) => false,
                    None => false,
                    _ => true
                })
                .cloned()
                .collect();
            
            let mut add_nodes = |conn: &Connection, is_self: bool| {
                let from = conn.from;
                let to = conn.to;
                while nodes.len() <= from || nodes.len() <= to {
                    nodes.push(None);
                }
                if is_self {
                    nodes[from] = self.nodes[from].clone();
                    nodes[to] = self.nodes[to].clone();
                } else {
                    nodes[from] = other.nodes[from].clone();
                    nodes[to] = other.nodes[to].clone();
                }
            };
            
            let mut conns = Vec::new();
            let bias = self.bias;
            
            let len = (self.conns.len() as f64).min(other.conns.len() as f64) as usize;

            for i in 0..len {
                let new_conn = match (&self.conns[i], &other.conns[i]) {
                    (Some(conn1), Some(conn2)) => {
                        if random::<f64>() < 0.5 {
                            add_nodes(conn1, true);
                            Some(conn1.clone())
                        } else {
                            add_nodes(conn2, false);
                            Some(conn2.clone())
                        }
                    },
                    (Some(conn), None) => {
                        add_nodes(conn, true);
                        Some(conn.clone())
                    },
                    _ => {
                        None
                    }
                };
                conns.push(new_conn);
            }

            for maybe_conn in self.conns.iter().skip(len) {
                if let Some(conn) = maybe_conn {
                    add_nodes(conn, true);
                    conns.push(Some((*conn).clone()));
                } else {
                    conns.push(None);
                }
            }

            Self {
                nodes,
                conns,
                bias,
                bio: self.bio
            }
        }

        fn add_connection(&mut self, neat: &mut Neat) {
            for _ in 0..100 {
                let from = randint(self.nodes.len());
                let to = randint(self.nodes.len());

                if let (Some(node1), Some(node2)) = (&self.nodes[from], &self.nodes[to]) {
                    if node1.can_connect_to(node2) && !neat.connections.contains(&(from, to)) {
                        neat.connect(self, from, to, random_weight());
                        break;
                    } else if node2.can_connect_to(node1) && !neat.connections.contains(&(to, from)) {
                        neat.connect(self, to, from, random_weight());
                        break;
                    }
                }
            }
        }

        fn add_node(&mut self, neat: &mut Neat) {
            if self.conns.len() == 0 {
                return;
            }
            for _ in 0..100 {
                let index = randint(self.conns.len());

                // Check if this connection exists in this genome
                if self.conns[index].is_none() {
                    continue;
                }

                if let Disabled = self.conns[index].as_ref().unwrap().state {
                    continue;
                }

                let connection = self.conns[index].as_ref().unwrap();
                let from = connection.from;
                let to = connection.to;
                let weight = connection.weight;

                // Check if the two nodes exist
                if let (Some(node1), Some(node2)) = (&self.nodes[from], &self.nodes[to]) {
                    let new_node = Node::new_hidden(node1, node2).expect("Wasn't able to create a new node between a connection for some ");
                    let new_index = neat.nodes;

                    // Check if 2 valid connections can be created with the new node
                    if !node1.can_connect_to(&new_node) || !new_node.can_connect_to(node2) {
                        continue;
                    }

                    // Check if any of the 2 connections already exists
                    if neat.connections.contains(&(from, new_index))
                    || neat.connections.contains(&(new_index, to)) {
                        continue;
                    }
                    
                    // Add the 2 connections
                    neat.connect(self, from, new_index, 1.);
                    neat.connect(self, new_index, to, weight);

                    self.conns[index].as_mut().unwrap().disable();

                    // Finally add the node and update the global counter
                    while self.nodes.len() < new_index {
                        self.nodes.push(None);
                    }
                    self.nodes.push(Some(new_node));
                    neat.nodes += 1;
                    break;
                } else {
                    panic!("Something is terribly wrong in 'add_node'!!");
                }
            }
        }

        fn mutate(mut self, neat: &mut Neat) -> Self {
            if random::<f64>() < neat.mutation_rate {
                if random::<f64>() < 0.01 {
                    self.bias = random_bias();
                }
                if random::<f64>() < 0.03 {
                    self.add_node(neat);
                }
                if random::<f64>() < 0.3 {
                    self.add_connection(neat);
                }
                if random::<f64>() < 0.01 {
                    self.bias *= 0.95;
                }
                for maybe_conn in &mut self.conns {
                    if let Some(conn) = maybe_conn.as_mut() {
                        if random::<f64>() < 0.1 {
                            conn.change_weight(random_weight());
                        }
                        if random::<f64>() < 0.1 {
                            conn.toggle();
                        }
                        if random::<f64>() < 0.1 {
                            conn.shift_weight();
                        }
                    }
                }
            }


            self
        }
    }

    use std::collections::HashSet;

    /// Neat struct that takes care of evolving the population based on the fitness scores
    #[derive(Debug)]
    pub struct Neat {
        nodes: usize,
        connections: HashSet<(usize, usize)>,
        genomes: Vec<Genome>,
        mutation_rate: f64
    }

    impl Neat {
        /// Creates a new population
        pub fn new(inputs: usize, outputs: usize, size: usize, mutation_rate: f64) -> Self {
            Self {
                nodes: 1 + inputs + outputs,
                connections: HashSet::new(),
                genomes: (0..size).map(|_| Genome::empty(inputs, outputs)).collect(),
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

        fn calculate_fitness(&self, calculate: fn(&Genome, bool) -> f64) -> Vec<f64> {
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
            scores.iter_mut().for_each(|score| *score = (*score * 100.) / total_score);
            scores
        }

        /// Takes care of evaluation, speciation, selection and mutation and creates the
        /// new population
        pub fn next_generation(&mut self, calculate: fn(&Genome, bool) -> f64) {
            let adjusted_score = self.calculate_fitness(calculate);
            let total_mean = 100. / (adjusted_score.len() as f64);

            let mut speciess = self.speciate();

            println!("Number of species = {}", speciess.len());

            let mut next_generation = Vec::new();

            for species in &mut speciess {
                species.sort_by(|&a, &b| adjusted_score[b].partial_cmp(&adjusted_score[a]).unwrap());
                // println!("{:?}", species.iter().map(|&i| adjusted_score[i]).collect::<Vec<_>>());

                let top = (0.6 * species.len() as f64).round() as usize;

                let num = species.iter().map(|&i| adjusted_score[i]).sum::<f64>();
                let num = (num / total_mean).round() as usize;
                for _ in 0..num {
                    let index1 = species[randint(top)];
                    let index2 = species[randint(top)];
                    let one = &self.genomes[index1];
                    let two = &self.genomes[index2];

                    let child = if adjusted_score[index1] > adjusted_score[index2] {
                        one.cross(two)
                    } else {
                        two.cross(one)
                    };

                    let child = child.mutate(self);

                    next_generation.push(child);
                }
            }

            self.genomes = next_generation;
        }

        fn connect(&mut self, genome: &mut Genome, from: usize, to: usize, weight: f64) {
            let innov_num = self.connections.len();
            self.connections.insert((from, to));

            while genome.conns.len() < innov_num {
                genome.conns.push(None);
            }

            let new_connection = Connection::new(from, to, weight);
            genome.conns.push(Some(new_connection));
        }
    }
}


#[cfg(test)]
mod tests {

    extern crate slow_nn;
    extern crate gym;

    use super::neat::Neat;
    use super::neat::Genome;
    use super::rand::random;

    // #[test]
    // fn test_adding_connections() {
    //     let mut neat = Neat::new(3, 2, 3, 0.1);
    //     let mut genome = Genome::empty(3, 2);
    //     for _ in 0..3 {
    //         genome.add_connection(&mut neat);
    //         println!("{:#?}", genome.conns);
    //     }
    //     println!("{:#?}", neat.connections);
    // }

    // #[test]
    // fn test_adding_nodes() {
    //     let mut neat = Neat::new(3, 2, 3, 0.1);
    //     let mut genome = Genome::empty(3, 2);
    //     for _ in 0..3 {
    //         genome.add_node(&mut neat);
    //         println!("{:#?}", genome.nodes);
    //         genome.add_connection(&mut neat);
    //         println!("{:#?}", genome.conns);
    //     }
    // }

    // #[test]
    // fn test_speciation() {
    //     let mut neat = Neat::new(3, 2, 10, 0.1);
    //     let mut genomes: Vec<_> = (0..1000).map(|_| Genome::empty(3, 2)).collect();

    //     for _ in 0..100 {
    //         for genome in &mut genomes {
    //             if random::<f64>() < 0.1 {
    //                 genome.add_connection(&mut neat);
    //             }
    //             if random::<f64>() < 0.01 {
    //                 genome.add_node(&mut neat);
    //             }
    //         }
    //     }

    //     neat.genomes = genomes;
    //     neat.speciate();
    // }

    use slow_nn::Network;
    use super::Connection;

    fn tanh(x: f64) -> f64 {
        x.tanh()
    }

    fn sigmoid(x: f64) -> f64 {
        1. / (1. + (-x).exp())
    }

    fn disp(genome: &Genome) {
        println!("Connections = {}, Hidden Nodes = {}", genome.conns.len(), genome.hidden());
    }

    fn calculate_xor(genome: &Genome, display: bool) -> f64 {
        let conns: Vec<_> = genome
            .conns
            .iter()
            .filter(|c| c.is_some() && c.as_ref().unwrap().is_enabled())
            .map(|c| {
                let c = c.as_ref().unwrap();
                (c.from, c.to, c.weight).into()
            })
            .collect();
        let net = Network::from_conns(genome.bias, 2, 1, genome.hidden() - 4, &conns);
        let mut fitness = 10000.0;
        let inputs = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
        let outputs = [0., 0., 0., 1.];
        let mut for_display = [0f64; 4];
        let mut index = 0;
        let mut diff = 0.;

        for (input, output) in inputs.iter().zip(outputs.iter()) {
            let out = net.predict(input, sigmoid);
            let mut pred: f64 = out[0];
            for_display[index] = pred;
            index += 1;
            diff += (pred - output).abs();
        }

        fitness -= diff;

        if display {
            // println!("Output = {:#?}", for_display);
            println!("Fitness = {}", fitness);
            println!("Connections = {}, Hidden Nodes = {}", genome.conns.len(), genome.hidden());
        }

        fitness
    }

    use gym::{GymClient, SpaceData::DISCRETE};

    fn calculate_cart(genome: &Genome, display: bool) -> f64 {
        let conns: Vec<_> = genome
            .conns
            .iter()
            .filter(|c| c.is_some() && c.as_ref().unwrap().is_enabled())
            .map(|c| {
                let c = c.as_ref().unwrap();
                (c.from, c.to, c.weight).into()
            })
            .collect();
        let net = Network::from_conns(genome.bias, 128, 6, genome.hidden() - 135, &conns);
        let mut fitness = 0.0;
        
        let client = GymClient::default();
        let env = client.make("SpaceInvaders-ram-v0");
        let mut input = [0f64; 128];

        let init = env.reset().unwrap().get_box().unwrap();
        for i in 0..128 {
            input[i] = init[i];
        }

        loop {
            let action = net.predict(&input, sigmoid);
            let mut max = action[0];
            let mut argmax = 0;
            for i in 1..6 {
                if action[i] > max {
                    max = action[i];
                    argmax = i;
                }
            }

            let action = DISCRETE(argmax);
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
            fitness += 1.0 + state.reward;
        }
        env.close();

        if display {
            // println!("Output = {:#?}", for_display);
            println!("Fitness = {}", fitness);
            println!("Connections = {}, Hidden Nodes = {}", genome.conns.len(), genome.hidden());
        }

        fitness
    }

    use std::{thread, time};

    #[test]
    fn test_neat() {
        let mut neat = Neat::new(128, 6, 50, 0.1);
        
        for gen in 1..=10 {
            println!("-------Gen #{}--------", gen);
            neat.next_generation(calculate_cart);
            thread::sleep(time::Duration::from_millis(200));
        }
    }
}
