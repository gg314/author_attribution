use std::fmt;

pub struct Iris {
    pub sepal_length: f64,
    pub sepal_width: f64,
    pub petal_length: f64,
    pub petal_width: f64,
    pub class: String
}

impl fmt::Display for Iris {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: sepals: ({}, {}) cm, petals: ({}, {}) cm", self.class, self.sepal_length, self.sepal_width, self.petal_length, self.petal_width)
    }
}