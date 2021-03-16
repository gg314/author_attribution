pub fn train(_pos: Vec<Vec<f64>>, _neg: Vec<Vec<f64>>) -> i32 {
    

    // while the optimality conditions are violated:

        // select q variables from the working set B. The remaining variables are fixed at their current value
        // decompose problem and solve QP-subproblem: Optimize W(alpha) on B
    
    // terminate and return alhpa

    return 1;
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