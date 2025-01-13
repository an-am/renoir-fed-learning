use burn::{
    module::Module, nn::{Linear, LinearConfig, Relu}, prelude::{Backend, *}, tensor::{backend::AutodiffBackend, Tensor},
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep}
};

use nn::{loss::MseLoss, LinearRecord};

use crate::dataset::ClientBatch;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    layer1: Linear<B>,
    layer2: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct ModelConfig;

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let layer1 = LinearConfig::new(9, 128)
            .with_bias(true)
            .init(device);
        let layer2 = LinearConfig::new(128, 1)
            .with_bias(true)
            .init(device);

        Model {
            layer1,
            layer2,
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, data: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.layer1.forward(data);
        let x = self.activation.forward(x);

        self.layer2.forward(x)
    }

    pub fn forward_step(&self, item: ClientBatch<B>) -> RegressionOutput<B> {
        let targets: Tensor<B, 2> = item.target.unsqueeze_dim(1);
        let output: Tensor<B, 2> = self.forward(item.data);

        let loss = MseLoss::new().forward(output.clone(), targets.clone(), nn::loss::Reduction::Mean);

        RegressionOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<ClientBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, item: ClientBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_step(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<ClientBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, item: ClientBatch<B>) -> RegressionOutput<B> {
        self.forward_step(item)
    }
}

impl<B: Backend> Clone for ModelRecord<B> {
    fn clone(&self) -> Self { 
        ModelRecord { 
            layer1: LinearRecord {
                weight: self.layer1.weight.clone(),
                bias: self.layer2.bias.clone(),
            },
            layer2: LinearRecord {
                weight: self.layer2.weight.clone(),
                bias: self.layer2.bias.clone(),
            }, 
            activation: self.activation.clone(),
        }
    }
}

