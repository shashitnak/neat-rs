
extern crate rand;
extern crate slow_nn;

use super::traits::*;
use super::random::*;

trait SparseInsert<T> {
    fn sparse_insert(&mut self, index: usize, val: T);
}

impl<T> SparseInsert<T> for Vec<Option<T>> {
    fn sparse_insert(&mut self, index: usize, val: T) {
        while self.len() <= index {
            self.push(None);
        }
        self[index] = Some(val);
    }
}

#[derive(Debug, Clone)]
enum Node {
    Input,
    Bias,
    Hidden(u128),
    Output
}

impl Node {
    fn bias() -> Self {
        Node::Bias
    }

    fn input() -> Self{
        Node::Input
    }

    fn output() -> Self {
        Node::Output
    }

    fn is_hidden(&self) -> bool {
        match *self {
            Node::Hidden(_) => true,
            _ => false
        }
    }

    fn value(&self) -> u128 {
        match *self {
            Node::Input | Node::Bias => 0,
            Node::Hidden(val) => val,
            Node::Output => std::u128::MAX
        }
    }

    fn can_connect_to(&self, to: &Self) -> bool {
        self.value() < to.value()
    }

    fn new_hidden(from: &Self, to: &Self) -> Option<Self> {
        let val1 = from.value();
        let val2 = to.value();
        
        let mid = val1 + (val2 - val1) / 2;
        
        if val1 < mid && mid < val2 {
            Some(Node::Hidden(mid))
        } else {
            None
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
    from: usize,
    to: usize,
    weight: f64,
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
pub struct Genotype {
    nodes: Vec<Option<Node>>,
    conns: Vec<Option<Connection>>,
    bias: f64,
    inputs: usize,
    outputs: usize
}

impl Genotype {
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

    fn change_bias(&mut self) {
        self.bias *= 95.;
    }

    fn new_bias(&mut self) {
        self.bias = random_bias();
    }

    fn add_connection<T: GlobalNeatCounter>(&mut self, neat: &mut T) {
        for _ in 0..100 {
            let from = randint(self.nodes.len());
            let to = randint(self.nodes.len());

            if let (Some(node1), Some(node2)) = (&self.nodes[from], &self.nodes[to]) {
                if node1.can_connect_to(node2) {
                    if let Some(innov) = neat.try_adding_connection(from, to) {
                        let new_connection = Connection::new(from, to, random_weight());
                        self.conns.sparse_insert(innov, new_connection);
                        break;
                    }
                } else if node2.can_connect_to(node1) {
                    if let Some(innov) = neat.try_adding_connection(to, from) {
                        let new_connection = Connection::new(to, from, random_weight());
                        self.conns.sparse_insert(innov, new_connection);
                        break;
                    }
                    break;
                }
            }
        }
    }

    fn add_node<T: GlobalNeatCounter>(&mut self, neat: &mut T) {
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
            // let weight = connection.weight;

            let node1 = self.nodes[from].as_ref()
                .expect("How can the node not exist when connection to this node does?");
            let node2 = self.nodes[to].as_ref()
                .expect("How can the node not exist when connection to this node does?");
            
            if let Some(new_node) = Node::new_hidden(node1, node2) {
                let new_index = neat.get_new_node();
                self.nodes.sparse_insert(new_index, new_node);

                let innov = neat.try_adding_connection(from, new_index)
                    .expect("How can this new node already have a connection?");
                let connection = Connection::new(from, new_index, random_weight());
                self.conns.sparse_insert(innov, connection);

                let innov = neat.try_adding_connection(new_index, to)
                    .expect("How can this new node already have a connection?");
                let connection = Connection::new(new_index, to, random_weight());
                self.conns.sparse_insert(innov, connection);

                self.conns[index].as_mut().unwrap().disable();

                break;
            }
        }
    }
}

impl Gene for Genotype {
    fn empty(inputs: usize, outputs: usize) -> Self {
        let nodes = (0..1).map(|_| Some(Node::bias()))
            .chain((0..inputs).map(|_| Some(Node::input())))
            .chain((0..outputs).map(|_| Some(Node::output())))
            .collect();
        Self {
            nodes,
            conns: Vec::new(),
            bias: random_bias(),
            inputs: inputs,
            outputs: outputs
        }
    }

    fn is_same_species_as(&self, other: &Self) -> bool {
        self.distance_from(other) < 3.
    }

    fn cross(&self, other: &Self) -> Self {
        let mut nodes: Vec<_> = self
            .nodes
            .iter()
            .take_while(|x| x.is_some() && !x.as_ref().unwrap().is_hidden())
            .cloned()
            .collect();
        
        let mut add_nodes = |conn: &Connection, is_self: bool| {
            let from = conn.from;
            let to = conn.to;
            if is_self {
                nodes.sparse_insert(from, self.nodes[from].clone().unwrap());
                nodes.sparse_insert(to, self.nodes[to].clone().unwrap());
            } else {
                nodes.sparse_insert(from, other.nodes[from].clone().unwrap());
                nodes.sparse_insert(to, other.nodes[to].clone().unwrap());
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
            inputs: self.inputs,
            outputs: self.outputs
        }
    }

    fn mutate<T: GlobalNeatCounter>(mut self, neat: &mut T) -> Self {
        match randint(100) {
            0..=2 => self.add_node(neat),
            3 => self.new_bias(),
            4 => self.change_bias(),
            5..=34 => self.add_connection(neat),
            34..=40 if self.conns.len() >= 1 => {
                let index = randint(self.conns.len());
                if let Some(connection) = self.conns[index].as_mut() {
                    match randint(100) {
                        0..=1 => connection.shift_weight(),
                        2..=3 => connection.change_weight(random_weight()),
                        _ => {}
                    }
                }
            }
            _ => {}
        }

        self
    }

    fn predict(&self, input: &[f64]) -> Vec<f64> {
        let connections: Vec<_> = self
            .conns
            .iter()
            .filter(|c| c.is_some())
            .map(|c| match c.as_ref() {
                Some(conns) => (conns.from, conns.to, conns.weight).into(),
                _ => panic!("this line will never be reached"),
            })
            .collect();
        
        let inputs = self.inputs;
        let outputs = self.outputs;
        let hidden = self.nodes.len() - 1 - inputs - outputs;
        
        let net = slow_nn::Network::from_conns(self.bias, inputs, outputs, hidden, &connections);
        net.predict(input, tanh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    struct Neat {
        connections: HashSet<(usize, usize)>,
        nodes: usize
    }

    impl Neat {
        fn new(inputs: usize, outputs: usize) -> Self {
            Self {
                connections: HashSet::new(),
                nodes: 1 + inputs + outputs
            }
        }
    }

    impl GlobalNeatCounter for Neat {
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

    #[test]
    fn test_node() {
        let input = Node::input();
        let output = Node::output();

        let hidden = Node::new_hidden(&input, &output).unwrap();
        let hidden1 = Node::new_hidden(&input, &hidden).unwrap();
        let hidden2 = Node::new_hidden(&hidden1, &output).unwrap();
        let hiddden3 = Node::new_hidden(&hidden1, &hidden).unwrap();
    }

    #[test]
    fn test_genome() {
        let mut genome1 = Genotype::empty(3, 2);
        let mut genome2 = Genotype::empty(3, 2);
        let mut neat = Neat::new(3, 2);

        for _ in 0..1000 {
            genome1 = genome1.mutate(&mut neat);
        }
    }
}