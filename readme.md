# wyrm

[![Crates.io badge](https://img.shields.io/crates/v/wyrm.svg)](https://crates.io/crates/wyrm)
[![Docs.rs badge](https://docs.rs/wyrm/badge.svg)](https://docs.rs/wyrm/)
[![Build Status](https://travis-ci.org/maciejkula/wyrm.svg?branch=master)](https://travis-ci.org/maciejkula/wyrm)

A reverse mode, define-by-run, low-overhead autodifferentiation library.

## Features

Performs backpropagation through arbitrary, define-by-run computation graphs,
emphasizing low overhead estimation of sparse, small models on the CPU.

Highlights:

1. Low overhead.
2. Built-in support for sparse gradients.
3. Define-by-run.
4. Trivial Hogwild-style parallelisation, scaling linearly with the number of CPU cores available.

## Quickstart

The following defines a univariate linear regression model, then
backpropagates through it.

```rust
let slope = ParameterNode::new(random_matrix(1, 1));
let intercept = ParameterNode::new(random_matrix(1, 1));

let x = InputNode::new(random_matrix(1, 1));
let y = InputNode::new(random_matrix(1, 1));

let y_hat = slope.clone() * x.clone() + intercept.clone();
let mut loss = (y.clone() - y_hat).square();
```

To optimize the parameters, create an optimizer object and
go through several epochs of learning:

```rust
let mut optimizer = SGD::new().learning_rate(0.1);

for _ in 0..num_epochs {
    let x_value: f32 = rand::random();
    let y_value = 3.0 * x_value + 5.0;

    // You can re-use the computation graph
    // by giving the input nodes new values.
    x.set_value(x_value);
    y.set_value(y_value);

    loss.forward();
    loss.backward(1.0);

    optimizer.step(loss.parameters());
}
```

You can use `rayon` to fit your model in parallel, by first creating a set of shared
parameters, then building a per-thread copy of the model:

```rust
let slope_param = Arc::new(HogwildParameter::new(random_matrix(1, 1)));
let intercept_param = Arc::new(HogwildParameter::new(random_matrix(1, 1)));
let num_epochs = 10;

(0..rayon::current_num_threads())
    .into_par_iter()
       .for_each(|_| {
           let slope = ParameterNode::shared(slope_param.clone());
           let intercept = ParameterNode::shared(intercept_param.clone());
           let x = InputNode::new(random_matrix(1, 1));
           let y = InputNode::new(random_matrix(1, 1));
           let y_hat = slope.clone() * x.clone() + intercept.clone();
           let mut loss = (y.clone() - y_hat).square();

           let optimizer = SGD::new().learning_rate(0.1);

           for _ in 0..num_epochs {
               let x_value: f32 = rand::random();
               let y_value = 3.0 * x_value + 5.0;

               x.set_value(x_value);
               y.set_value(y_value);

               loss.forward();
               loss.backward(1.0);

               optimizer.step(loss.parameters());
           }
       });
```

### BLAS support
You should enable BLAS support to get (much) better performance out of matrix-multiplication-heavy
workloads. To do so, add the following to your `Cargo.toml`:

```
ndarray = { version = "0.11.0", features = ["blas", "serde-1"] }
blas-src = { version = "0.2.0", default-features = false, features = ["openblas"] }
openblas-src = { version = "0.6.0", default-features = false, features = ["cblas"] }
```

### Fast numerics

Enable the `fast-math` option to use fast approximations to transcendental functions.
This should give substantial speed gains in networks that are `exp`, `ln`, or `tanh`-heavy.

License: MIT
