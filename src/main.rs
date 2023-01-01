mod apd;
mod adamax;

use crate::apd::Distribution;
use crate::apd::RNG;
use crate::adamax::AdaMax;

fn score(w: f32) -> f32 {
    const SOLUTION: f32 = 0.5;
    (w - SOLUTION).powf(2.0)
}

fn main() {
    let mut rng = RNG::new(123);
    let dist = apd::Gaussian::new(0.0, 1.0);
    let npop = 50;
    let sigma = 0.1;
    let mut optim = AdaMax::new(0.05);
    let mut w = dist.sample(&mut rng);
    let mut score_weights = 0.0;
    let mut data_log: Vec<(i32, f32)> = Vec::new();
    
    for i in 0..300 {
        let mut sum_returns = 0.0;
        for x in 0..npop {
            let eps = dist.sample(&mut rng);
            let w_t = w + sigma * eps;
            sum_returns = sum_returns + score(w_t) * eps;
        }
        w = optim.update(w, sum_returns/(npop as f32 * sigma));
        score_weights = score(w);
        data_log.push((i, score_weights));
        println!("Generation {}: Score - {}, Weights: {}", i, score_weights, w);
    }
}