pub struct AdaMax {
    alpha: f32,
    beta1: f32,
    beta2: f32,
    m: f32,
    u: f32,
    t: f32
}

impl AdaMax {
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha,
            beta1: 0.9,
            beta2: 0.999,
            m: 0.0,
            u: 0.0,
            t: 0.0
        }
    }

    pub fn update(&mut self, param: f32, gradient: f32) -> f32 {
        const EPSILON: f32 = 0.000001;
        self.t += 1.0;
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gradient;
        self.u = f32::max(self.beta2 * self.u, gradient.abs());
        param - (self.alpha * self.m)/((1.0 - self.beta1.powf(self.t)) * (self.u + EPSILON))
    }
}