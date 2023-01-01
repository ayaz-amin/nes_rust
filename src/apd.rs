use std::f32::consts::TAU;

pub struct RNG {
    seed: u32,
    pos: u32
}

impl RNG {
    pub fn new(seed: u32) -> Self { Self {seed, pos: 1} }
    
    fn sample(&mut self) -> f32 {
        const NOISE1: u32 = 0x68E31DA4;
        const NOISE2: u32 = 0xB5297A4D;
        const NOISE3: u32 = 0x1B56C4E9;
        
        let mut mangled = self.pos as u32;
        mangled = mangled.wrapping_mul(NOISE1);
        mangled = mangled.wrapping_add(self.seed);
        mangled ^= mangled >> 8;
        mangled = mangled.wrapping_add(NOISE2);
        mangled ^= mangled << 8;
        mangled = mangled.wrapping_mul(NOISE3);
        mangled ^= mangled >> 8;
        self.pos += 1;
        (mangled as f32) / (u32::MAX as f32)
    }
}

pub trait Distribution {
    type Data;
    fn log_prob(&self, x: Self::Data) -> f32;
    fn sample(&self, rng: &mut RNG) -> Self::Data;
}

pub struct Gaussian {
    mean: f32,
    stddev: f32
}

impl Gaussian {
    pub fn new(mean: f32, stddev: f32) -> Self {
        Self {mean, stddev}
    }
}

impl Distribution for Gaussian {
    type Data = f32;
    fn log_prob(&self, x: Self::Data) -> f32 {
        const HALF_LN_TAU: f32 = 0.91893853320467274178032973640562;
        let z = (x - self.mean) / self.stddev;
        -self.stddev.ln() - HALF_LN_TAU - 0.5 * z.powf(2.0)
    }

    fn sample(&self, rng: &mut RNG) -> Self::Data {
        let u1: f32 = rng.sample();
        let u2: f32 = rng.sample();
        let mag: f32 = self.stddev * (-2.0 * u1.ln()).sqrt();
        mag * (TAU * u2).cos() + self.mean
    }
}

pub struct Categorical {
    probs: Vec<f32>
}

impl Categorical {
    pub fn new(mut probs: Vec<f32>) -> Self {
        let sum: f32 = probs.iter().sum();
        if sum > 1.0 {
            probs.iter_mut().for_each(|x| *x /= sum);
        }
        Self {probs}
    }
}

impl Distribution for Categorical {
    type Data = usize;
    fn log_prob(&self, x: Self::Data) -> f32 {
        let sum: f32 = self.probs.iter().sum();
        let prob = self.probs[x];
        if sum > 1.0 {
            return (prob / sum).ln();
        }
        return prob.ln();
    }

    fn sample(&self, rng: &mut RNG) -> Self::Data {
        let mut noise: f32 = rng.sample();
        for i in 0..self.probs.len() {
            if noise < self.probs[i as usize] {
                return i as usize;
            }
            noise -= self.probs[i as usize];
        }
        unreachable!("Loop should return");
    }
}