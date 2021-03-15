use std::ops::{Mul, Add};
use std::iter::Sum;
use rand::Rng;
use std::fmt;


#[derive(Debug)]
pub struct Sample {
    pub values: Vec<f64>,
    pub class: i8
}

impl fmt::Display for Sample {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{");
        let mut n = 0;
        for v in &self.values {
            if n == 0 {
                ();
            } else if (n == 1) {
                write!(f, "{}", v);
            } else {
                write!(f, ", {}", v);
            }
            n += 1;
        };
        write!(f, "}}, ")
    }
}


fn dot<T>(v1: &Vec<T>, v2: &Vec<T>) -> T
where 
    T: Mul<Output = T> + Copy + Sum
 {
    assert_eq!(v1.len(), v2.len());
    return v1.iter().zip(v2.iter()).map(|(x, y)| *x * *y).sum();
}

fn sum<T>(v1: &Vec<T>, v2: &Vec<T>) -> Vec<T>
where 
    T: Add<Output = T> + Copy + Sum
 {
    assert_eq!(v1.len(), v2.len());
    return v1.iter().zip(v2.iter()).map(|(x, y)| *x + *y).collect();
}


fn norm(vin: &Vec<i16>) -> Vec<f64> {
    let mut sum = 0;
    for i in vin {
        sum += i*i;
    }

    let sum = (sum as f64).sqrt();
    let mut vout = Vec::new();
    for i in vin {
        vout.push( *i as f64 / sum )
    }

    return vout
}

fn update_weights(weights: &Vec<f64>, rate: f64, xj: &Sample, yj: i8) -> Vec<f64> {

    // let mut output: Vec<f64> = Vec::new();
    let mut temp: Vec<f64> = Vec::new();
    
    for i in &xj.values {
        temp.push( rate * ((xj.class - yj) as f64) * i )
    }

    let output = sum(&temp, &weights); 

    return output;
}


fn f(w: &Vec<f64>, xj: &Sample) -> i8 {
    if dot(&w, &xj.values) > 0.0 {
        return 1;
    } else {
        return 0;
    }
}

pub fn train(samples: &Vec<Sample>) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut weights = Vec::new();
    let rate = 0.01;
    let max_epochs = 100;
    
    let sample = samples.get(0).unwrap();
    for _i in &sample.values {
        weights.push(rng.gen_range(0.0..0.01))
    }

    /*
    println!("\nClass 1:");
    for s in samples {
        if s.class == 1 {
            print!("{:}", s);
        }
    }
    println!("\n\nClass 0:");
    for s in samples {
        if s.class != 1 {
            print!("{:}", s);
        }
    }
    println!("\n");
    */

    let mut epoch = 1;
    while epoch < max_epochs {
        let mut errors = 0;
        for s in samples {
            let yj = f(&weights, &s);
            weights = update_weights(&weights, rate, &s, yj);
            if (s.class - yj) != 0 {
                errors += 1;
            }
        }
        if errors == 0 {
            println!("Perceptron converged with weights: {:?} after {} epochs.", weights, epoch);
            break;
        }
        epoch += 1;
    }

    if epoch >= max_epochs {
        println!("Perceptron did not converge! Data may not be linearly-separable, but you can try increasing `max_epochs`, training rate, etc.");
        println!("Perceptron weights after {} epochs: {:?}", epoch, weights);
    }

    return weights
}


pub fn test(weights: Vec<f64>, samples: &Vec<Sample>) {
    let mut successes = 0;
    let mut total = 0;
    for s in samples {
        if f(&weights, &s) == s.class {
            successes += 1;
        }
        total += 1;
    }

    let percentage = format!("{:.1}%", 100.0* successes as f64 / total as f64);
    println!("Success rate: {}/{} ({})", successes, total, percentage)
    
}