# Federated Learning with Renoir in Rust

Renoir is a **distributed data processing platform**, 
based on the **dataflow paradigm**, that provides an ergonomic programming interface, similar to that of Apache Flink, but has **much better performance** characteristics. More infos on Renoir [here](https://databrush.it/renoir/overview/).

A **fixed set** of federated clients train their local neural network on their own dataset, satisfying **data locality** principles.
A **server** aggregates new model weights, following the **Federated Averaging** algorithm, for a total of 10 rounds.
